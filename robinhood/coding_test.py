class coding_test:
    def findShortestSubArray(self, nums):
        left, right, dict = {}, {}, {}
        for i, n in enumerate(nums):
            dict[n] = dict.get(n, 0) + 1
            if n not in left:
                left[n] = i
            right[n] = i

        max_val = max(dict.values())
        min_len = len(nums)
        for n, freq in dict.items():
            if max_val == dict[n]:
                if min_len > right[n] - left[n]:
                    min_len = right[n] - left[n] + 1
        return min_len

    # def calculateProfit(self, transactions):
    def rectangular(self, n):
        ans = [['*'] * n for _ in range(n)]
        for i in range(1, n-1):
            for j in range(1, n-1):
                ans[i][j] = ''
        return ans

    def most_frequent_digit(self, input):
        all = ''.join(map(str, input))
        cnt = {}

        max_freq = 0
        max_idx = ''
        for d in all:
            cnt[d] = cnt.get(d, 0) + 1
            if max_freq < cnt[d]:
                max_freq = cnt[d]
                max_idx = d
        return [int(max_idx), cnt[max_idx]]

    def reverse_digit(self, s):
        if not s:
            return s

        n = len(s)
        hn = n // 2
        ans = ['a'] * n
        for i in range(hn):
            ans[2 * i + 1] = s[2 * i]
            ans[2 * i] = s[2 * i + 1]
        if n % 2 == 1:
            ans[-1] = s[-1]
        return ''.join(ans)

    def cut_wood(self, wood, k):
        def is_valid_cut(wood, cur_len, k):
            pieces = 0
            for w in wood:
                pieces += w // cur_len
            return True if pieces >= k else False

        max_len = max(wood)
        left, right = 1, max_len    # left, right is length
        while left + 1 < right:
            mid = (left + right) // 2
            if is_valid_cut(wood, mid, k):
                left = mid
            else:
                right = mid - 1
        if is_valid_cut(wood, right, k):
            return right
        elif is_valid_cut(wood, left, k):
            return left
        else:
            return 0

    def evenDigitsNumber(self, array):
        for i, e in enumerate(array):
            if len(str(e)) % 2 == 0:
                return i

    def binaryPatternMatching(self, matrix, k):
        if not matrix:return

        def get_local_sum(matrix, i, j, k):
            sum = 0
            for n in range(k):
                for m in range(k):
                    sum += matrix[i+n][j+m]
            return sum

        n = len(matrix)
        if not matrix[0]:return
        m = len(matrix[0])

        w = m - k + 1
        sum_matr = [0] * (w * w)
        cnt = 0
        for i in range(w):
            for j in range(w):
                sum_matr[cnt] = get_local_sum(matrix, i, j, k)
                cnt += 1

        max_val = max(sum_matr)
        midx = [i for i, x in enumerate(sum_matr) if x == max_val]

        ans = []
        for idx in midx:
            cidx = idx % k
            ridx = idx // k
            for n in range(k):
                for m in range(k):
                    ans.append(matrix[ridx+n][cidx+m])
        return sum(list(set(ans)))

    def reverse(self, x: int) -> int:
        if not x:
            return 0

        xs = str(x)
        if len(xs) == 1:
            return x

        while xs[-1] == '0':
            xs = xs[:-1]

        if xs[0] == '-':
            ans = int('-' + xs[1:][::-1])
        else:
            ans = int(xs[::-1])

        if ans > (1 << 31) - 1 or ans < -(1 << 31):
            return 0
        else:
            return ans


if __name__ == '__main__':
    ct = coding_test()
    print(ct.reverse(1463847412))
