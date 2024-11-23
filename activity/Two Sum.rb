def two_sum(nums, target)
  hash_map = {}
  nums.each_with_index do |num, i|
    complement = target - num
    if hash_map[complement]
      return [hash_map[complement], i]
    end
    hash_map[num] = i
  end
  []
end

# Example usage
nums = [2, 7, 11, 15]
target = 9
puts two_sum(nums, target).inspect 
