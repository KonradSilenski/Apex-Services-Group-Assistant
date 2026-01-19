# (full file)
class MatchContext:
    def __init__(self):
        self.added_codes = set()
        self.boosts = {}
        self.excluded_codes = set()
        self.logs = []

    def add_code(self, code, reason=""):
        self.added_codes.add(code)
        if reason:
            self.logs.append(f"Added {code}: {reason}")

    def boost(self, target, amount, reason=""):
        self.boosts[target] = self.boosts.get(target, 0) + amount
        if reason:
            self.logs.append(f"Boosted {target} by {amount}: {reason}")

    def exclude_code(self, code, reason=""):
        self.excluded_codes.add(code)
        if reason:
            self.logs.append(f"Excluded {code}: {reason}")
