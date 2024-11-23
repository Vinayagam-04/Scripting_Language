def is_palindrome(x)
  return false if x < 0
  x.to_s == x.to_s.reverse
end

# Example usage
x = 121
puts is_palindrome(x)  # Output: true
