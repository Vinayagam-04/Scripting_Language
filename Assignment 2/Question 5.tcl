#!/usr/bin/env tclsh

# Create the main window
set mainWindow [tk::toplevel .]
wm title $mainWindow "List Box Example"

# Create a frame to hold the list boxes and buttons
frame .frame1
pack .frame1 -padx 10 -pady 10

# Create the first list box
listbox .frame1.list1 -height 10 -width 30
pack .frame1.list1 -side left -padx 10

# Create the second list box
listbox .frame1.list2 -height 10 -width 30
pack .frame1.list2 -side right -padx 10

# Add sample items to the first list box
foreach item {Apple Banana Cherry Date Fig Grape} {
    .frame1.list1 insert end $item
}

# Function to handle item selection from the first list box
proc addItem {} {
    # Get the selected item from the first list box
    set selected [.frame1.list1 curselection]
    if {$selected ne ""} {
        # Get the item text
        set item [.frame1.list1 get $selected]
        # Insert the item into the second list box if not already present
        if {[.frame1.list2 see $item] == ""} {
            .frame1.list2 insert end $item
        }
    }
}

# Function to remove the selected item from the second list box
proc removeItem {} {
    set selected [.frame1.list2 curselection]
    if {$selected ne ""} {
        # Remove the selected item from the second list box
        .frame1.list2 delete $selected
    }
}

# Button to add item from list1 to list2
button .frame1.addButton -text "Add to List 2" -command addItem
pack .frame1.addButton -side left -padx 10

# Button to remove selected item from list2
button .frame1.removeButton -text "Remove from List 2" -command removeItem
pack .frame1.removeButton -side left -padx 10

# Start the Tk main loop
tk::mainloop
