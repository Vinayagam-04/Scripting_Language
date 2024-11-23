def reverse(x)
  if x < 0
    reversed = -x.to_s.reverse.to_i
  else
    reversed = x.to_s.reverse.to_i
  end
  return 0 if reversed > 2**31 - 1 || reversed < -2**31
  reversed
end

# Example usage
x = -123
puts reverse(x)
