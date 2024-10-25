class BankAccount
  attr_reader :balance
  
  def initialize
    @balance = 0
  end

  # Method to add money to the account
  def add_balance(amount)
    if amount > 0
      @balance += amount
      puts "Successfully deposited $#{amount}. Current balance: $#{@balance}."
    else
      puts "Error: Deposit amount must be positive."
    end
  end

  # Method to withdraw money from the account
  def withdraw(amount)
    if amount > 0 && amount <= @balance
      @balance -= amount
      puts "Successfully withdrew $#{amount}. Current balance: $#{@balance}."
    elsif amount > @balance
      puts "Error: Insufficient balance."
    else
      puts "Error: Withdrawal amount must be positive."
    end
  end

  # Method to check the current balance
  def check_balance
    puts "Current balance: $#{@balance}."
  end
end

# Example usage:
account = BankAccount.new
account.add_balance(100)
account.withdraw(30)
account.check_balance
