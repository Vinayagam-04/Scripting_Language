# Function to calculate the volume of a sphere
proc sphereVolume {radius} {
    # Calculate the volume using the formula
    set volume [expr {(4.0 / 3.0) * pi * ($radius ^ 3)}]
    return $volume
}

# Main program
# Prompt the user for the radius of the sphere
puts "Please enter the radius of the sphere:"
flush stdout  ;# Ensures that the prompt is displayed immediately

# Get the user input
set radius [gets stdin]

# Validate input (ensure it's a number)
if {![regexp {^[0-9]*\.?[0-9]*$} $radius]} {
    puts "Error: Please enter a valid number for the radius."
} else {
    # Call the function to calculate the volume
    set volume [sphereVolume $radius]

    # Display the volume
    puts "The volume of the sphere with radius $radius is: $volume"
}
