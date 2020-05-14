class Brain:
    def get(self, s):
        return

    def add_observation(self, s, a, r, s_, batch_size):
        return

    def learn(self):
        return

    @staticmethod
    def get_available(s):
        used, hand_card, penalty, last = s[0], s[2], s[4], None
        available = []

        for i in range(1, len(used) + 1):
            last = used[-i]
            if last is not None:
                break

        for item in hand_card:
            if last is None:
                available.append(item)
            else:
                if last.valid_card(item, penalty):
                    available.append(item)

        return available
