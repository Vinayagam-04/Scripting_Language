# Parent class (superclass) for all vehicles
class Vehicle
  attr_accessor :mileage, :fuel_level

  def initialize(mileage, fuel_level)
    @mileage = mileage
    @fuel_level = fuel_level
  end

  # Common functionality for all vehicles
  def calculate_mileage(distance)
    @mileage += distance
    puts "Mileage updated. Current mileage: #{@mileage} km."
  end

  def check_fuel_level
    puts "Current fuel level: #{@fuel_level} liters."
  end

  def perform_maintenance
    puts "Performing general vehicle maintenance."
  end
end

# Subclass for Car, inheriting from Vehicle
class Car < Vehicle
  attr_accessor :number_of_doors

  def initialize(mileage, fuel_level, number_of_doors)
    super(mileage, fuel_level)  # Calls Vehicle's initialize method
    @number_of_doors = number_of_doors
  end

  # Car-specific behavior
  def play_music
    puts "Playing music in the car."
  end
end

# Subclass for Truck, inheriting from Vehicle
class Truck < Vehicle
  attr_accessor :cargo_capacity

  def initialize(mileage, fuel_level, cargo_capacity)
    super(mileage, fuel_level)  # Calls Vehicle's initialize method
    @cargo_capacity = cargo_capacity
  end

  # Truck-specific behavior
  def load_cargo(weight)
    puts "Loading #{weight} tons of cargo. Capacity remaining: #{@cargo_capacity - weight} tons."
  end
end

# Subclass for Motorcycle, inheriting from Vehicle
class Motorcycle < Vehicle
  attr_accessor :helmet_included

  def initialize(mileage, fuel_level, helmet_included)
    super(mileage, fuel_level)  # Calls Vehicle's initialize method
    @helmet_included = helmet_included
  end

  # Motorcycle-specific behavior
  def perform_wheelie
    puts "Performing a wheelie on the motorcycle!"
  end
end

# Example usage
car = Car.new(5000, 50, 4)
car.calculate_mileage(100)
car.play_music
car.check_fuel_level

truck = Truck.new(20000, 80, 15)
truck.calculate_mileage(200)
truck.load_cargo(5)

motorcycle = Motorcycle.new(3000, 10, true)
motorcycle.calculate_mileage(50)
motorcycle.perform_wheelie
