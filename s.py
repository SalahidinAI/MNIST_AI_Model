def maxSubArray(nums):
    res = 0
    right = len(nums) - 1
    left = 0
    while left <= right:
        s = sum(nums[left: right + 1])
        if s > res: res = s
        print(nums[left: right + 1])
        if nums[left] > nums[right]: right -= 1
        elif nums[left] > nums[right]: left += 1
        else:
            r = right
            l = left
            while l <= r:
                el1 = nums[l]
                el2 = nums[r]
                if r == l: left += 1
                if el1 == el2:
                    l += 1
                    r -= 1
                elif el1 < el2:
                    left += 1
                    break
                else:
                    right -= 1
                    break
    return res


# print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4])) # 6
print(maxSubArray([1,2,-1,-2,2,1,-2,1,4,-5,4])) # 6
