def longest_common_prefix(strs)
  return "" if strs.empty?
  prefix = strs[0]
  strs.each do |str|
    while str.index(prefix) != 0
      prefix = prefix[0...prefix.length-1]
      return "" if prefix.empty?
    end
  end
  prefix
end

# Example usage
strs = ["flower", "flow", "flight"]
puts longest_common_prefix(strs)  # Output: "fl"
