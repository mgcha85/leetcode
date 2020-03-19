
class Solution:
    # Given an array of integers, return indices of the two numbers such that they add up to a specific target.
    #
    # You may assume that each input would have exactly one solution, and you may not use the same element twice.
    #
    # Example:
    #
    # Given nums = [2, 7, 11, 15], target = 9,
    #
    # Because nums[0] + nums[1] = 2 + 7 = 9,
    # return [0, 1].

    def two_sum(self, arr, target):
        N = len(arr)
        M = float(N)
        comb = [-1] * int(M * (M - 1) / 2)
        cnt = 0
        for i in range(N):
            for j in range(N):
                if j > i:
                    comb[cnt] = (i, j)
                    cnt += 1

        idx = []
        for c in comb:
            sum = arr[c[0]] + arr[c[1]]
            if sum == target:
                idx += list(c)
        return idx

    def run_two_sum(self):
        arr = [2, 5, 7, 11, 14, 15]
        target = 7
        idx = self.two_sum(arr, target)
        print(idx)

    # You are given two non-empty linked lists representing two non-negative integers. The digits are stored in
    # reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
    #
    # You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    #
    # Example:
    #
    # Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    # Output: 7 -> 0 -> 8
    # Explanation: 342 + 465 = 807.

    def addTwoNumbers(self):
        class Node:
            def __init__(self, value=None, next=None):
                self.value = value
                self.next = next

            def __str__(self):
                return str(self.value)

        def print_list(node):
            while node:
                print(node),
                node = node.next

        n1_ = Node(2)
        n2_ = Node(4)
        n3_ = Node(3)

        n1_.next = n2_
        n2_.next = n3_

        n1 = Node(5)
        n2 = Node(6)
        n3 = Node(4)

        n1.next = n2
        n2.next = n3

        quotient = 0

        nodes = []
        for i, (f, s) in enumerate(zip([n3_, n2_, n1_], [n3, n2, n1])):
            sum = f.value + s.value
            remainder = sum % 10
            nodes.append(Node(remainder + quotient))
            quotient = sum // 10
            if i > 0:
                nodes[-2].next = nodes[-1]
        print_list(nodes[0])

    def longest_substring(self, input):
        """
        Given a string, find the length of the longest substring without repeating characters.

        Example 1:

        Input: "abcabcbb"
        Output: 3
        Explanation: The answer is "abc", with the length of 3.
        Example 2:

        Input: "bbbbb"
        Output: 1
        Explanation: The answer is "b", with the length of 1.
        Example 3:

        Input: "pwwkew"
        Output: 3
        Explanation: The answer is "wke", with the length of 3.
                     Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
        """
        ch = set(input)
        N = len(ch)
        if N == 1:
            return N

        L = len(input)
        while True:
            for i in range(L):
                substring = set(input[i:i + N])
                l = len(substring)
                if N == l:
                    return l, input[i:i + N]
            N -= 1

    def median_of_two_sorted_array(self, A, B):
        """
        There are two sorted arrays nums1 and nums2 of size m and n respectively.

        Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

        You may assume nums1 and nums2 cannot be both empty.

        Example 1:

        nums1 = [1, 3]
        nums2 = [2]

        The median is 2.0
        Example 2:

        nums1 = [1, 2]
        nums2 = [3, 4]

        The median is (2 + 3)/2 = 2.5
        """

        nA = len(A)
        nB = len(B)
        N = nA + nB
        nM = (N + 1) // 2

        ia, ib = 0, 0
        tot = 0
        pval = A[0]
        while ia < nA and ib < nB:
            if A[ia] < B[ib]:
                val = A[ia]
                ia += 1
            else:
                val = B[ib]
                ib += 1
            tot += 1
            if N % 2 != 0 and nM <= tot:
                return val
            elif nM < tot:
                return (val + pval) / 2
            pval = val

    def longest_palindromic_substring(self, input):
        """
        Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

        Example 1:

        Input: "babad"
        Output: "bab"
        Note: "aba" is also a valid answer.
        Example 2:

        Input: "cbbd"
        Output: "bb"
        """
        def search_closest_ch(input, ch):
            idx = 1
            while idx < len(input):
                if input[idx] == ch:
                    return idx
                idx += 1
            return -1

        def is_palindrom(input):
            N = (len(input) + 1) // 2
            for i in range(N):
                if input[i] != input[-i - 1]:
                    return False
            return True

        L = len(input)
        M = (L + 1) // 2
        for i in range(M):
            lidx = search_closest_ch(input[i:], input[i])
            if lidx < 0:
                continue
            substring = input[i: i + lidx + 1]
            if is_palindrom(substring):
                return substring

    def zigzag_conversion(self, str, numRows):
        """
        The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

        P   A   H   N
        A P L S I I G
        Y   I   R
        And then read line by line: "PAHNAPLSIIGYIR"

        Write the code that will take a string and make this conversion given a number of rows:

        string convert(string s, int numRows);
        Example 1:

        Input: s = "PAYPALISHIRING", numRows = 3
        Output: "PAHNAPLSIIGYIR"
        Example 2:

        Input: s = "PAYPALISHIRING", numRows = 4
        Output: "PINALSIGYAHRPI"
        Explanation:

        P     I    N
        A   L S  I G
        Y A   H R
        P     I
        """

        row, col = 1, 0
        zz_str = {0: {0: str[0]}, }

        reverse = False
        for s in str[1:]:
            if row in zz_str:
                zz_str[row][col] = s
            else:
                zz_str[row] = {col: s}
            if row == 0 or row == numRows - 1:
                reverse = ~reverse

            if reverse:
                row -= 1
                col += 1
            else:
                row += 1

        out_str = ''
        for ridx in sorted(list(zz_str.keys())):
            row = zz_str[ridx]
            for cidx in sorted(list(row.keys())):
                col = row[cidx]
                out_str += col
        return out_str

    def reverse_integer(self, input):
        """"
        Given a 32-bit signed integer, reverse digits of an integer.

        Example 1:

        Input: 123
        Output: 321
        Example 2:

        Input: -123
        Output: -321
        Example 3:

        Input: 120
        Output: 21
        """
        sign = ''
        if input < 0:
            input = -input
            sign = '-'

        return int(sign + str(input)[::-1])

    def string_to_integer(self, str):
        """
        Implement atoi which converts a string to an integer.

        The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

        The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

        If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

        If no valid conversion could be performed, a zero value is returned.

        Note:

        Only the space character ' ' is considered as whitespace character.
        Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. If the numerical value is out of the range of representable values, INT_MAX (231 − 1) or INT_MIN (−231) is returned.
        Example 1:

        Input: "42"
        Output: 42
        Example 2:

        Input: "   -42"
        Output: -42
        Explanation: The first non-whitespace character is '-', which is the minus sign.
                     Then take as many numerical digits as possible, which gets 42.
        Example 3:

        Input: "4193 with words"
        Output: 4193
        Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.
        Example 4:

        Input: "words and 987"
        Output: 0
        Explanation: The first non-whitespace character is 'w', which is not a numerical
                     digit or a +/- sign. Therefore no valid conversion could be performed.
        Example 5:

        Input: "-91283472332"
        Output: -2147483648
        Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer.
                     Thefore INT_MIN (−231) is returned.
        """

        str = str.replace(' ', '')
        num_set = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

        init_ch = str[0]
        if init_ch == '-':
            str = str[1:]
            sign = False
        elif init_ch in num_set:
            sign = True
        else:
            return 0

        conv = []
        out = 0
        for i, s in enumerate(str):
            if s in num_set:
                num = num_set[s]
            else:
                continue
            conv.append(num)

        N = len(conv)
        for i, c in enumerate(conv):
            out += c * (10 ** (N - i - 1))

        limit = (1 << 31)
        if not(-limit <= out <= limit - 1):
            if sign:
                return limit
            else:
                return -limit

        return out

    def palindrome_number(self, x):
        """
        Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

        Example 1:

        Input: 121
        Output: true
        Example 2:

        Input: -121
        Output: false
        Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
        Example 3:

        Input: 10
        Output: false
        Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
        Follow up:

        Coud you solve it without converting the integer to a string?
        """

        if x < 0:
            return False

        import math
        L = int(math.log10(x))

        contents = []
        for i in range(L, 0, -1):
            q = x // (10 ** i)
            x = x - q * (10 ** i)
            contents.append(q)
        contents.append(x)

        N = len(contents)
        for i in range(N // 2):
            if contents[i] != contents[N - i - 1]:
                return False
        return True

    def regex_matching(self, s, p):
        """
        Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.

        '.' Matches any single character.
        '*' Matches zero or more of the preceding element.
        The matching should cover the entire input string (not partial).

        Note:

        s could be empty and contains only lowercase letters a-z.
        p could be empty and contains only lowercase letters a-z, and characters like . or *.
        Example 1:

        Input:
        s = "aa"
        p = "a"
        Output: false
        Explanation: "a" does not match the entire string "aa".
        Example 2:

        Input:
        s = "aa"
        p = "a*"
        Output: true
        Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
        Example 3:

        Input:
        s = "ab"
        p = ".*"
        Output: true
        Explanation: ".*" means "zero or more (*) of any character (.)".
        Example 4:

        Input:
        s = "aab"
        p = "c*a*b"
        Output: true
        Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".
        Example 5:

        Input:
        s = "mississippi"
        p = "mis*is*p*."
        Output: false
        """
        if '.' not in p and '*' not in p and s != p:
            return False

        P = set(p)
        S = set(s)

        if s.replace('.', '') == '':
            if len(s) != len(p):
                return False

        if '.' in p and '*' in p:
            p = p.replace('*', '.')

        an = ord('a')
        zn = ord('z')

        n = [len(s), len(p)]
        if n[1] > n[0]:
            maxi = 1
            mini = 0
        else:
            maxi = 0
            mini = 1

        # filter s and p
        for i in range(n[maxi]):
            if maxi == 0:
                if i < mini:
                    if not (an <= ord(p[i]) <= zn or p[i] in '.*'):
                        return False
                if not(an <= ord(s[i]) <= zn):
                    return False
            else:
                if i < mini:
                    if not (an <= ord(s[i]) <= zn):
                        return False
                if not (an <= ord(p[i]) <= zn or p[i] in '.*'):
                    return False

        if '.' in p:
            ndot = p.count('.')
            ss = S - P
            nss = len(ss)
            if nss == ndot:
                p = p.replace('.', ss.pop())
            else:
                return False

        if '*' in p:
            comp = S - P
            if len(comp) > 0:
                return False

            ps = p.split('*')
            for q in ps:
                s = s.replace(q, '')
            if len(s) > 0:
                return False

        return True

    def container_with_most_water(self, arr):
        """
        Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
        Note: You may not slant the container and n is at least 2.

        Example:

        Input: [1,8,6,2,5,4,8,3,7]
        Output: 49
        """

        contents = {}
        for i, a in enumerate(arr):
            contents[i] = a

        N = len(arr)

        comb = []
        for i in range(N):
            for j in range(i):
                comb.append((i, j))

        output = []
        for c in comb:
            x1, x2 = c
            y = min([contents[x1], contents[x2]])
            output.append((x1 - x2) * y)
        return max(output)

    def interger_to_roman(self, input):
        """
        Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

        Symbol       Value
        I             1
        V             5
        X             10
        L             50
        C             100
        D             500
        M             1000
        For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

        Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

        I can be placed before V (5) and X (10) to make 4 and 9.
        X can be placed before L (50) and C (100) to make 40 and 90.
        C can be placed before D (500) and M (1000) to make 400 and 900.
        Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.

        Example 1:

        Input: 3
        Output: "III"
        Example 2:

        Input: 4
        Output: "IV"
        Example 3:

        Input: 9
        Output: "IX"
        Example 4:

        Input: 58
        Output: "LVIII"
        Explanation: L = 50, V = 5, III = 3.
        Example 5:

        Input: 1994
        Output: "MCMXCIV"
        Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
        """
        from math import log10
        def get_closeset_idx(numbers, input):
            N = len(numbers)
            for i in range(N - 1):
                if numbers[i] <= input < numbers[i + 1]:
                    return i
            return N - 1

        symbols = 'IVXLCDM'
        numbers = [1, 5, 10, 50, 100, 500, 1000]

        output = ''
        while input > 0:
            idx = get_closeset_idx(numbers, input)
            sin = str(input)
            n = len(sin) - 1
            q = int(sin[0])

            if q == 4:
                val = q * (10 ** n)
                eidx = get_closeset_idx(numbers, val) + 1
                sidx = get_closeset_idx(numbers, 10 ** n)
                output += symbols[sidx] + symbols[eidx]
                input -= val
                continue
            elif q == 9:
                val = q * (10 ** n)
                eidx = get_closeset_idx(numbers, val) + 1
                sidx = get_closeset_idx(numbers, 10 ** n)
                output += symbols[sidx] + symbols[eidx]
                input -= val
                continue
            else:
                output += symbols[idx]
                input -= numbers[idx]

            if input < numbers[idx]:
                idx -= 1
        return output

    def three_sum(self, arr):
        """
        15.3Sum
        Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

        Note:

        The solution set must not contain duplicate triplets.

        Example:

        Given array nums = [-1, 0, 1, 2, -1, -4],

        A solution set is:
        [
          [-1, 0, 1],
          [-1, -1, 2]
        ]

        """

        N = len(arr)
        C = 3
        c = 3

        nom, den = 1, 1
        for x in range(N, N - C, -1):
            nom *= x
            den *= c
            c -= 1

        comb = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    comb.append([i, j, k])

        out = []
        for c in comb:
            val = []
            for x in c:
                val.append(arr[x])
            val = sorted(val)
            if sum(val) == 0 and val not in out:
                out.append(val)
        return out

    def three_sum_closest(self, arr, target):
        """
        16.3Sum Closest
        Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

        Example:

        Given array nums = [-1, 2, 1, -4], and target = 1.

        The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

        """
        def sum(arr, idx):
            sum = 0
            for i in idx:
                sum += arr[i]
            return sum

        def sub_arr(arr, idx):
            out = []
            for i in idx:
                out.append(arr[i])
            return out

        def get_closest(arr, target):
            idx = 1 << 31
            for i, a in enumerate(arr):
                diff = abs(a - target)
                if diff < idx:
                    idx = i
            return idx

        N = len(arr)
        idx = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    idx.append([i, j, k])

        sums = []
        for i in idx:
            sums.append(sum(arr, i))

        cidx = get_closest(sums, target)
        return sub_arr(arr, idx[cidx])

    def letter_comb(self, input):
        """
        Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.
        A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

        Example:
        Input: "23"
        Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
        Note:

        Although the above answer is in lexicographical order, your answer could be in any order you want.
        """
        from itertools import product
        map = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

        letters = []
        for i in input:
            letters.append(map[i])

        out = []
        for pair in list(product(*letters)):
            out.append(''.join(pair))
        return out

    def four_sum(self, arr, target):
        """
        Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
        Note:
        The solution set must not contain duplicate quadruplets.
        Example:
        Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.
        A solution set is:
        [
          [-1,  0, 0, 1],
          [-2, -1, 1, 2],
          [-2,  0, 0, 2]
        ]
        """
        from itertools import combinations
        candidates = combinations(arr, 4)

        output = []
        for cand in candidates:
            if sum(cand) == target:
                output.append(sorted(cand))
        return output

    def remove_nth_node(self, arr, n):
        """
        Given a linked list, remove the n-th node from the end of list and return its head.
        Example:
        Given linked list: 1->2->3->4->5, and n = 2.
        After removing the second node from the end, the linked list becomes 1->2->3->5.
        Note:
        Given n will always be valid.
        Follow up:
        Could you do this in one pass?
        """
        class node:
            def __init__(self, num, node=None, pnode=None):
                self.num = num
                self.next = node
                self.prev = pnode

        pn = node(arr[0])
        init = pn
        for a in arr[1:]:
            no = node(a)
            pn.next = no
            no.prev = pn
            pn = no

        for i in range(n-1):
            rn = no.prev

        rn.prev.next = rn.next
        rn.next.prev = rn.prev
        del rn

        while True:
            print(init.num)
            if init.next is None:
                break
            init = init.next

    def generate_parentheses(self, n=3):
        """
        Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

        For example, given n = 3, a solution set is:

        [
          "((()))",
          "(()())",
          "(())()",
          "()(())",
          "()()()"
        ]

        """
        from itertools import permutations

        class array:
            def __init__(self, arr):
                self.arr = arr

            def shuffle(self, idx):
                out = [1]
                for i in idx:
                    out.append(self.arr[i])
                return out

        dict = {1: '(', -1: ')'}
        def check_paren(arr):
            sum = 0
            for a in arr:
                sum += a
                if sum < 0:
                    return False
            return True

        arr = [1] * n + [-1] * n
        next_arr = array(arr[1:])

        narr_perm = permutations(next_arr.arr)
        narr_perm = set(list(narr_perm))

        out_list = []
        for np in narr_perm:
            if check_paren([1, *np]):
                out_list.append([1, *np])

        ret = []
        for out in out_list:
            ele = ''
            for o in out:
                ele += dict[o]
            ret.append(ele)
        return ret

    def swap_nodes_in_pairs(self, n):
        """
        Given a linked list, swap every two adjacent nodes and return its head.

        You may not modify the values in the list's nodes, only nodes itself may be changed.

        Example:
        Given 1->2->3->4, you should return the list as 2->1->4->3.
        """

        if n % 2 != 0:
            print('bye')
            return

        class Node:
            def __init__(self, num, next=None):
                self.num = num
                self.next = next

        def swap(nodeList):
            depth = len(nodeList) - 1
            for i in range(0, depth, 2):
                pnode = nodeList[i]
                nnode = nodeList[i + 1]
                nnode.next = pnode
                pnode.next = None
                nodeList[i] = nnode
                nodeList[i + 1] = pnode
            return nodeList

        def merge(nodeList):
            depth = len(nodeList)
            for i in range(1, depth, 4):
                pnode = nodeList[2 * i - 1]
                nnode = nodeList[2 * i]
                pnode.next = nnode
                nodeList[2 * i - 1] = pnode
                nodeList[2 * i] = nnode
            return nodeList[0]

        pnode = Node(1)
        nodes = [pnode]
        for i in range(2, n + 1):
            node = Node(i)
            pnode.next = node
            nodes.append(node)

        nodes = swap(nodes)
        offset = merge(nodes)

        while True:
            print(offset.num)
            if offset.next is None:
                break
            offset = offset.next

    def divide_two_int(self, dividend, divisor):
        """
        Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.
        Return the quotient after dividing dividend by divisor.
        The integer division should truncate toward zero.

        Example 1:
        Input: dividend = 10, divisor = 3
        Output: 3
        Example 2:

        Input: dividend = 7, divisor = -3
        Output: -2
        Note:
        Both dividend and divisor will be 32-bit signed integers.
        The divisor will never be 0.
        Assume we are dealing with an environment which could only store integers within the 32-bit signed integer
        range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 231 − 1 when the division result overflows.

        """
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """

        if dividend == 0:
            return

        if dividend > 0 and divisor > 0:
            sign = 1
        elif dividend < 0 and divisor < 0:
            sign = 1
            dividend = -dividend
            divisor = -divisor
        elif dividend < 0:
            sign = -1
            dividend = -dividend
        else:
            sign = -1
            divisor = -divisor

        of = (1 << 31)
        if divisor == 1:
            if not(-of <= dividend < of - 1):
                if sign > 0:
                    return sign * (of - 1)
                else:
                    return sign * of
            return sign * dividend

        res = 0
        i = 0
        while divisor << i <= dividend:
            i += 1

        for j in range(i, -1, -1):
            div = (divisor << j)
            if div <= dividend:
                dividend -= div
                res += 1 << j

        if res <= -1 << 31:
            return -res

        return sign * res

    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """

        def is_max(A):
            B = sorted(A)[::-1]
            for a, b in zip(A, B):
                if a != b:
                    return False
            return True

        def bigger_closest(A, val):
            diff = 1 << 31
            ret = A[0]
            idx = 0
            for i, a in enumerate(A):
                if a > val:
                    d = abs(a - val)
                    if d < diff:
                        ret = a
                        idx = i
            return ret, idx

        def swap(A):
            n = len(A) // 2
            for i in range(n):
                temp = A[i]
                A[i] = A[-i - 1]
                A[-i - 1] = temp

        if is_max(nums) is True:
            swap(nums)
            return

        if nums[-1] > nums[-2]:
            temp = nums[-1]
            nums[-1] = nums[-2]
            nums[-2] = temp
            return

        N = len(nums)
        for i in range(N - 3, -1, -1):
            if is_max(nums[i:]) is False:
                n, nidx = bigger_closest(nums[i + 1:], nums[i])

                remains = []
                for j, a in enumerate(nums[i:]):
                    if nidx + 1 != j:
                        remains.append(a)

                nums[i] = n
                nums[i + 1:] = sorted(remains)
                break

    def search(self, nums, target, mid=0):
        N = len(nums)

        if N < 1:
            return -1

        if N == 1:
            if nums[0] == target:
                return 0
            else:
                return -1

        mid = N // 2

        g1 = nums[:mid]
        g2 = nums[mid:]
        self.search(g1, target, 0)
        self.search(g2, target, mid)

        i1 = 0
        i2 = 0
        while i1 < len(g1) and i2 < len(g2):
            if g1[i1] == target:
                return i1 + mid
            if g2[i2] == target:
                return i2 + mid

            i1 += 1
            i2 += 1
        return -1

    def isValidSudoku(self, board):
        def get_column(board, ncol):
            for row in board:
                yield row[ncol]

        def get_batch(board):
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    out = []
                    for r in board[i: i + 3]:
                        out.extend(r[j: j + 3])
                    yield out

        board_batch = get_batch(board)
        # 3 x 3 check
        for bb in board_batch:
            bb_str = ''.join(bb).replace('.', '')
            n = len(bb_str)
            if n != len(set(bb_str)):
                return False

        for i in range(9):
            row = board[i]
            row_str = ''.join(row).replace('.', '')
            n = len(row_str)
            if n != len(set(row_str)):
                return False

        for i in range(9):
            col = get_column(board, i)
            col_str = ''.join(col).replace('.', '')
            n = len(col_str)
            if n != len(set(col_str)):
                return False

        return True

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        from itertools import combinations
        out = []

        n = len(candidates)
        for i in range(n):
            for cb in combinations(candidates, i + 1):
                sumc = sum(cb)
                if sumc == target:
                    out.append(list(cb))

                elif sumc < target:
                    diff = target - sumc
                    mult = [x for x in cb if diff % x == 0]
                    if mult:
                        q = diff // mult[0]
                        out.append(list(cb) + [mult[0]] * q)
        return out

    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        class Node:
            def __init__(self, val, prev=None, next=None):
                self.val = val
                self.next = next
                self.prev = prev

        l1_init = l1
        l2_init = l2

        pnode = None
        while l1 and l2:
            if l1.val < l2.val:
                ln = l1
                l1.next.prev = None
            else:
                ln = l2
                l2.next.prev = None

            if pnode is None:
                node = Node(ln.val)
            else:
                pnode.next = node

            pnode = node
            l1 = l1.next
            l2 = l2.next
        return node

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """

        if len(nums) <= 1:
            return nums

        while True:
            try:
                idx = nums.index(val)
                nums.pop(idx)
            except:
                return nums

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """

        M = len(needle)
        N = len(haystack)
        if M > N:
            return -1
        elif haystack == needle:
            return 0

        for i in range(N - M + 1):
            print(haystack[i: i + M])
            if needle == haystack[i: i + M]:
                return i
        return -1

    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """

        if n == 1:
            return '1'
        elif n >= 30:
            return

        strn = '1'
        for i in range(1, n):
            ps = strn[0]
            contents = [[ps, 1]]

            for j in range(1, len(strn)):
                if ps == strn[j]:
                    contents[-1][-1] += 1
                else:
                    contents.append([strn[j], 1])
                ps = strn[j]

            strn = ''
            for con in contents:
                strn += str(con[-1])
                strn += con[0]

        return strn

    def isPalindrome(self, s):
            """
            :type s: str
            :rtype: bool
            """

            s = s.replace(' ', '')
            if len(s) <= 0:
                return True

            def trim(s):
                for ch in s:
                    if not ch.isalpha():
                        s = s.replace(ch, "")
                    else:
                        s = s.lower()
                return s

            s = trim(s)
            N = len(s)
            if N == 0: return True

            m = N // 2

            for i in range(m):
                if s[i] != s[-i - 1]:
                    return False
            return True

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        while nums:
            ele = nums.pop(0)
            if ele not in nums:
                return ele
            else:
                nums.remove(ele)
        return ele

    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n == 0:
            return

        if n <= 26:
            res = [n]
        else:
            res = []
            while n > 26:
                r = n % 26
                n = n // 26
                if r == 0:
                    n -= 1
                    r = 26
                res.append(r)
            res.append(n)

        ret = ''
        for r in res[::-1]:
            ret += chr(64 + r)
        return ret

    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """

        def get_five(n):
            if n < 5:
                return 0
            q = n // 5
            return q + get_five(q)
        return get_five(n)

    def rotate(self, nums, k):
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        if n == 0 or n < k:
            return

        tail = nums[-k:]
        header = nums[:-k]
        nums[:k] = tail
        nums[k:] = header
        return nums

    def reverseBits(self, n):
        b = bin(n)[2:]
        n = len(b)
        if n >= 32:
            return b[::-1]
        zp = '{number:0{width}d}'.format(width=32 - n, number=0)
        return int((zp + b)[::-1], 2)

    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        return len(bin(n)[2:].replace('0', ''))

    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """

        def get_happy_number(n):
            sum = 0
            for d in str(n):
                sum += int(d) ** 2
            return sum

        history = []
        while True:
            n = get_happy_number(n)
            if n == 1:
                return True
            elif n in history:
                return False
            else:
                history.append(n)

    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """

        def isPrime(n):
            # This is checked so that we can skip
            # middle five numbers in below loop
            if (n % 2 == 0 or n % 3 == 0):
                return False

            i = 5
            while (i * i <= n):
                if (n % i == 0 or n % (i + 2) == 0):
                    return False
                i = i + 6

            return True

        if n <= 1:
            return 0
        elif n == 2:
            return 1
        elif n < 5:
            return 2

        cnt = 3
        for i in range(5, n):
            if isPrime(i):
                cnt += 1
        return cnt

    def findShortestSubArray(self, nums):
        right, left, count = {}, {}, {}
        for i, n in enumerate(nums):
            if n not in left: left[n] = i
            right[n] = i
            count[n] = count.get(n, 0) + 1

        N = len(nums)
        degree = max(count.values())
        ans = N
        for ele, cnt in count.items():
            if cnt == degree:
                ans = min(ans, right[ele] - left[ele] + 1)
        return ans

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []

        m, n = len(matrix), len(matrix[0])
        if n == 0:
            return []

        seen = [[False] * n for _ in range(m)]
        ans = []
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r = c = di = 0

        for _ in range(m * n):
            ans.append(matrix[r][c])
            seen[r][c] = True
            cr, cc = r + dr[di], c + dc[di]
            if seen[r][c] is False and 0 <= cr < m and 0 <= cc < n:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r, c = r + dr[di], c + dc[di]
        return ans

    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """

        if n == 1:
            return [[1]]

        ans = [[0] * n for _ in range(n)]
        r = c = di = 0
        rd = [0, 1, 0, -1]
        cd = [1, 0, -1, 0]

        for i in range(n ** 2):
            ans[r][c] = i + 1
            cr, cc = r + rd[di], c + cd[di]
            if 0 <= cr < n and 0 <= cc < n and ans[cr][cc] == 0:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r, c = r + rd[di], c + cd[di]
        return ans

    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """

        rows = {0: []}
        wpl = []
        cnt = nrows = sum_len = 0
        for i, word in enumerate(words):
            sum_len += len(word)
            rows[nrows].append(word)

            cnt += 1
            if sum_len + cnt - 1 > maxWidth:
                sum_len = len(word)
                rows[nrows].pop(-1)
                nrows += 1
                rows[nrows] = [word]
                wpl.append(cnt - 1)
                cnt = 1

        ans = []
        for idx, words in rows.items():
            sum_len = 0
            n = len(words)

            for i, word in enumerate(words):
                if i < len(words) - 1:
                    words[i] = word + ' '
                sum_len += len(words[i])

            if idx == nrows:
                pad = (maxWidth - sum_len)
                words[-1] += ' ' * pad
            else:
                if n == 1:
                    words[-1] = words[-1] + ' ' * (maxWidth - sum_len)
                else:
                    pad = (maxWidth - sum_len) // (n - 1)
                    for k in range(n - 1):
                        words[k] = words[k] + ' ' * pad

                    width = sum_len + pad * (n - 1)
                    if maxWidth != width:
                        diff = maxWidth - width
                        for d in range(diff):
                            words[d] = words[d] + ' '
            ans.append(''.join(words))
        return ans

    def evaluate_expr(self, stack):
        res = stack.pop() if stack else 0

        # Evaluate the expression till we get corresponding ')'
        while stack and stack[-1] != ')':
            sign = stack.pop()
            if sign == '+':
                res += stack.pop()
            else:
                res -= stack.pop()
        return res

    def calculate(self, s: str) -> int:
        stack = []
        n, operand = 0, 0

        for i in range(len(s) - 1, -1, -1):
            ch = s[i]
            if ch.isdigit():

                # Forming the operand - in reverse order.
                operand = (10 ** n * int(ch)) + operand
                n += 1

            elif ch != " ":
                if n:
                    # Save the operand on the stack
                    # As we encounter some non-digit.
                    stack.append(operand)
                    n, operand = 0, 0

                if ch == '(':
                    res = self.evaluate_expr(stack)
                    stack.pop()

                    # Append the evaluated result to the stack.
                    # This result could be of a sub-expression within the parenthesis.
                    stack.append(res)

                # For other non-digits just push onto the stack.
                else:
                    stack.append(ch)

        # Push the last operand to stack, if any.
        if n:
            stack.append(operand)

        # Evaluate any left overs in the stack.
        return self.evaluate_expr(stack)

    def longestPalindrome(self, s: str) -> str:
        def is_palindrome(s):
            n = len(s)
            if n == 1:
                return True

            for i in range(n // 2):
                if s[i] != s[-i - 1]:
                    return False
            return True

        n = len(s)
        if n <= 1:
            return s
        if is_palindrome(s):
            return s

        end = len(s) - 1
        for j in range(end, 0, -1):
            for k in range(n - j + 1):
                if is_palindrome(s[k: k + j]):
                    return s[k: k + j]
        return ''

    def rob(self, nums):
        if len(nums) == 0:
            return 0
        if len(nums) <= 2:
            return max(nums)

        nums = [0, 0] + nums
        for i, num in enumerate(nums[2:]):
            nums[i + 2] = max(nums[i] + nums[i + 2], nums[i + 1])
        return max(nums)


if __name__ == '__main__':
    s = Solution()
    print(s.rob([2,7,9,3,1]))
