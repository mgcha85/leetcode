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

    def maxProfit1(self, prices):
        if not prices:
            return []

        if len(prices) == 1:
            return 0

        T_i10 = 0
        T_i11 = float('-inf')

        for price in prices:
            T_i10 = max(T_i10, T_i11 + price)
            T_i11 = max(T_i11, -price)
        return T_i10

    def maxProfit2(self, prices):
        T_ik0, T_ik1 = 0, float('-inf')

        for price in prices:
            T_ik0_old = T_ik0
            T_ik0 = max(T_ik0, T_ik1 + price)
            T_ik1 = max(T_ik1, T_ik0_old - price)
        return T_ik0

    def maxProfit3(self, prices):
        T_i10 = T_i20 = 0
        T_i11 = T_i21 = float('-inf')

        for price in prices:
            T_i20 = max(T_i20, T_i21 + price)
            T_i21 = max(T_i21, T_i10 - price)
            T_i10 = max(T_i10, T_i11 + price)
            T_i11 = max(T_i11, -price)
        return T_i20

    def maxProfit4(self, k, prices):
        if k > len(prices):
            T_ik0, T_ik1 = 0, float('-inf')
            for price in prices:
                T_ik0_old = T_ik0
                T_ik0 = max(T_ik0, T_ik1 + price)
                T_ik1 = max(T_ik1, T_ik0_old - price)
            return T_ik0

        T_ik0 = [0] * (k + 1)
        T_ik1 = [float('-inf')] * (k + 1)
        for price in prices:
            for i in range(k, 0, -1):
                T_ik0[i] = max(T_ik0[i], T_ik1[i] + price)
                T_ik1[i] = max(T_ik1[i], T_ik0[i - 1] - price)
        return T_ik0[k]

    def maxProfit5(self, prices):
        T_ik0_pre, T_ik0, T_ik1 = 0, 0, float('-inf')

        for price in prices:
            T_ik0_old = T_ik0
            T_ik0 = max(T_ik0, T_ik1 + price)
            T_ik1 = max(T_ik1, T_ik0_pre - price)
            T_ik0_pre = T_ik0_old
        return T_ik0

    def maxProfit6(self, prices, fee):
        T_ik0, T_ik1 = 0, float('-inf')

        for price in prices:
            T_ik0_old = T_ik0
            T_ik0 = max(T_ik0, T_ik1 + price - fee)
            T_ik1 = max(T_ik1, T_ik0_old - price)
        return T_ik0

    def findKthNumber(self, n: int, k: int) -> int:
        if n < 10:
            return k

        cur = 1
        k -= 1
        while k > 0:
            step = 0
            first = cur
            last = cur + 1

            while first <= n:
                step += min(n + 1, last) - first
                first *= 10
                last *= 10

            if step <= k:
                cur += 1
                k -= step
            else:
                cur *= 10
                k -= 1
        return cur

    def find_all(self, a, sub):
        start = 0
        while True:
            start = a.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub)

    def lastSubstring(self, s: str) -> str:
        if len(set(s)) == 1:
            return s

        max_s = sorted(list(s))[-1]
        idxs = list(self.find_all(s, max_s))

        while True:
            next = []
            for idx in idxs:
                if idx + 1 < len(s):
                    next.append(max_s + s[idx + len(max_s)])

            if not next:
                return s[idxs[0]:]

            max_s = sorted(next)[-1]
            idxs = list(self.find_all(s, max_s))
            if len(idxs) == 1:
                return s[idxs[0]:]

    def findDiagonalOrder(self, matrix):
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 0
        if n == 0:
            return []

        result = [0 for _ in range(m * n)]
        up = True
        row = col = 0

        for i in range(m * n):
            result[i] = matrix[row][col]

            # check right bolder before top bolder in up trend
            if up:
                if col == n - 1:
                    row = row + 1
                    up = not up
                elif row == 0:
                    col = col + 1
                    up = not up
                else:
                    row = row - 1
                    col = col + 1

            # check bottom bolder before left bolder in down trend
            else:
                if row == m - 1:
                    col = col + 1
                    up = not up
                elif col == 0:
                    row = row + 1
                    up = not up
                else:
                    row = row + 1
                    col = col - 1

        return result

    def shortest_distance(self):
        import sqlite3

        con = sqlite3.connect('leetcode.sqlite')
        c = con.cursor()
        sql = "SELECT min(abs(p1.x-p2.x)) as shortest FROM point p1 JOIN point p2 ON p1.x!=p2.x"
        return c.execute(sql).fetchall()

    def biggest_single(self):
        import sqlite3

        con = sqlite3.connect('leetcode.sqlite')
        c = con.cursor()
        sql = "SELECT num FROM my_numbers GROUP BY num HAVING COUNT(*)==1 ORDER BY num DESC LIMIT 1"
        return c.execute(sql).fetchall()

    def prefixString(self, a, b):
        pre_fix = []
        s = ''
        for e in a:
            s += e
            pre_fix.append(s)

        if len(set(b) - set(pre_fix)) == 0:
            return True
        else:
            return False

    def frameGenerator(self, n):
        ans = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == 0 or j == 0 or i == n-1 or j == n-1:
                    row.append('*')
                else:
                    row.append(' ')
            ans.append(row)
        return ans

    def sortMatrixByOccurrences(self, matrix):
        if not matrix:
            return
        if not matrix[0]:
            return

        m = len(matrix)
        freq = {}
        for r in range(m):
            for c in range(m):
                freq[matrix[r][c]] = freq.get(matrix[r][c], 0) + 1

        freq = sorted(freq.items(), key=lambda x: (x[1], x[0]))
        r = c = m-1
        sr, sc = m-1, m-1
        dr, dc = -1, 1
        for ele in freq:
            v, f = ele
            for i in range(f):
                matrix[r][c] = v
                cr, cc = r + dr, c + dc
                if not(0 <= cr < m) or not(0 <= cc < m):
                    sc -= 1
                    if sc < 0:
                        sr -= 1
                        sc = 0
                    r, c = sr, sc
                else:
                    r, c = cr, cc
        return matrix

    def sql(self):
        import sqlite3

        # con = sqlite3.connect('leetcode.sqlite')
        # sql = "CREATE TABLE orders (" \
        # "order_number INTEGER," \
        # "customer_number INTEGER,"\
        # "order_date DATE,"\
        # "required_date DATE,"\
        # "shipped_date DATE,"\
        # "status STRING,"\
        # "comment STRING" \
        # ");"
        # con.cursor().execute(sql)

        con = sqlite3.connect('leetcode.sqlite')
        c = con.cursor()
        sql = "SELECT customer_number FROM orders GROUP BY customer_number ORDER BY COUNT(*) DESC LIMIT 1"
        return c.execute(sql).fetchall()


if __name__ == '__main__':
    ct = coding_test()
    print(ct.sql())
    # print(ct.findDiagonalOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]))
