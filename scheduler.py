"""
Study Room Scheduler - the three AI algorithms (no web code in this file).

Problem: a student wants to book a study room. There are several choices
(room + time). The booking system says "free" or "busy", but its report is
often out of date. Which choice is best, and should the student book now or wait?

Three algorithms from three different course topics work as one chain:
  1. Hill climbing (local search)        -> finds the best choice
  2. Bayes' theorem (probability)        -> how sure are we that the room is really free?
  3. MDP, one Bellman update (decisions) -> Book now or Wait?

How they are connected:
  hill climbing -> the score of the best choice becomes the reward in the MDP
  Bayes         -> the probability "room is free" is used by the MDP
  MDP           -> uses both and makes the final decision

Run in the terminal:  python scheduler.py
"""
from dataclasses import dataclass, asdict

NAMES = ["Room 101, 09:00", "Room 101, 11:00", "Room 204, 13:00",
         "Room 204, 15:00", "Room 305, 17:00"]
DEFAULT_SCORES = [4, 6, 9, 7, 3]


@dataclass
class Settings:
    """All numbers of the model. They are assumptions (invented for this project)."""
    prior: float = 0.5                  # before the report: room is really free in 50% of cases
    p_free_if_available: float = 0.9    # system says "free" when the room is really free
    p_free_if_taken: float = 0.3        # system says "free" but the room is taken (old information)
    reward_book_taken: float = -2       # we book a room that is already taken
    reward_wait: float = -1             # waiting costs a little
    p_still_available: float = 0.6      # after waiting, a free room is still free
    p_becomes_available: float = 0.2    # after waiting, a taken room becomes free (cancellation)
    gamma: float = 0.9                  # later is worth a bit less than now

    def validate(self):
        for name in ("prior", "p_free_if_available", "p_free_if_taken",
                     "p_still_available", "p_becomes_available", "gamma"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number between 0 and 1")
        for name in ("reward_book_taken", "reward_wait"):
            if not isinstance(getattr(self, name), (int, float)):
                raise ValueError(f"{name} must be a number")


def check_scores(scores):
    if not isinstance(scores, (list, tuple)) or len(scores) < 2:
        raise ValueError("scores must be a list of at least 2 numbers")
    for s in scores:
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise ValueError("every score must be a number")


def check_report(report):
    if report not in ("free", "busy"):
        raise ValueError("report must be 'free' or 'busy'")


# ======================================================== ALGORITHM 1: HILL CLIMBING
def hill_climbing(scores, start=0):
    """Stand on one choice, look at the two neighbours, step to the better one.
    Stop when no neighbour is better. Returns the list of visited positions."""
    check_scores(scores)
    position = start
    path = [position]
    while True:
        neighbours = [p for p in (position - 1, position + 1) if 0 <= p < len(scores)]
        best_neighbour = max(neighbours, key=lambda p: scores[p])
        if scores[best_neighbour] > scores[position]:
            position = best_neighbour
            path.append(position)
        else:
            return path


def find_best(scores, restarts=False):
    """Run hill climbing from position 0. With restarts=True run it from every
    position and keep the best result (this fixes 'stuck on a small peak')."""
    check_scores(scores)
    if not restarts:
        return hill_climbing(scores, 0)
    paths = [hill_climbing(scores, start) for start in range(len(scores))]
    return max(paths, key=lambda path: scores[path[-1]])   # first best path wins a tie


# ======================================================== ALGORITHM 2: BAYES' THEOREM
def bayes(report, s):
    """P(available | report) = P(report | available) * P(available) / P(report).
    Returns (probability the room is really available, P(report))."""
    check_report(report)
    if report == "free":
        like_available, like_taken = s.p_free_if_available, s.p_free_if_taken
    else:
        like_available, like_taken = 1 - s.p_free_if_available, 1 - s.p_free_if_taken
    p_report = like_available * s.prior + like_taken * (1 - s.prior)
    if p_report == 0:
        raise ValueError("this report is impossible with these settings")
    return like_available * s.prior / p_report, p_report


# ======================================================== ALGORITHM 3: MDP (one Bellman update)
def mdp(p_available, best_score, s):
    """Two states (room free / taken), two actions (Book now / Wait).
    Bellman idea: value = reward now + gamma * value of what comes next.
    We are not sure about the state, so we average with the probability from Bayes."""
    book_if_free = best_score
    book_if_taken = s.reward_book_taken
    wait_if_free = s.reward_wait + s.gamma * (s.p_still_available * best_score)
    wait_if_taken = s.reward_wait + s.gamma * (s.p_becomes_available * best_score)

    book = p_available * book_if_free + (1 - p_available) * book_if_taken
    wait = p_available * wait_if_free + (1 - p_available) * wait_if_taken
    return {"action": "Book now" if book > wait else "Wait", "book": book, "wait": wait,
            "book_if_free": book_if_free, "book_if_taken": book_if_taken,
            "wait_if_free": wait_if_free, "wait_if_taken": wait_if_taken}


# ======================================================== THE WHOLE PROJECT
def decide(report, scores=None, settings=None, restarts=False, names=None):
    """Run the three steps in order and return everything (used by the web page and tests)."""
    scores = list(DEFAULT_SCORES if scores is None else scores)
    s = settings or Settings()
    s.validate()
    check_scores(scores)
    names = names or (NAMES if len(scores) == len(NAMES) else [f"Choice {i + 1}" for i in range(len(scores))])

    path = find_best(scores, restarts)                      # step 1
    best_index = path[-1]
    best_score = scores[best_index]
    p_available, p_report = bayes(report, s)                # step 2
    result = mdp(p_available, best_score, s)                # step 3
    return {"report": report, "names": names, "scores": scores, "path": path,
            "best_index": best_index, "best_name": names[best_index], "best_score": best_score,
            "prior": s.prior, "p_report": p_report, "p_available": p_available,
            "mdp": result, "action": result["action"], "restarts": restarts,
            "settings": asdict(s)}


def evaluate(scores=None, settings=None, restarts=False):
    """Average reward of three strategies (exact probability, no randomness).
    Higher is better."""
    scores = list(DEFAULT_SCORES if scores is None else scores)
    s = settings or Settings()
    s.validate()
    best_score = scores[find_best(scores, restarts)[-1]]
    without_report = mdp(s.prior, best_score, s)             # we only know the prior
    ours = 0.0
    for report in ("free", "busy"):
        try:
            p_available, p_report = bayes(report, s)
        except ValueError:                                   # this report can never happen
            continue
        r = mdp(p_available, best_score, s)
        ours += p_report * max(r["book"], r["wait"])          # use the report, choose the better action
    return {"always_book": without_report["book"], "always_wait": without_report["wait"], "ours": ours}


if __name__ == "__main__":
    for rep in ("free", "busy"):
        d = decide(rep)
        print(f"\n=== The system says: {rep.upper()} ===")
        print("1 Hill climbing :", " -> ".join(str(d['scores'][i]) for i in d["path"]),
              f"| best = {d['best_name']} (score {d['best_score']})")
        print(f"2 Bayes         : P(room is really free) = {d['p_available']:.3f}")
        print(f"3 MDP           : Book now = {d['mdp']['book']:.3f}, Wait = {d['mdp']['wait']:.3f}")
        print("RESULT          :", d["action"])
    e = evaluate()
    print("\n=== Evaluation (average reward, higher is better) ===")
    print(f"Always book now : {e['always_book']:.2f}")
    print(f"Always wait     : {e['always_wait']:.2f}")
    print(f"Our method      : {e['ours']:.2f}")
