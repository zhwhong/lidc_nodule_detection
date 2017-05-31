class CandidateFilter(object):
    def __init__(self, image):
        pass

    def _filter(self, x, y, h, w):
        return True

    def filter(self, candidates):
        post_filter = []
        for candidate in candidates:
            # print(candidate)
            if self._filter(candidate[0], candidate[1], candidate[2],
                            candidate[3]):
                continue
            post_filter.append(candidate)
        return post_filter
