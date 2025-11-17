from virtual_mouse import run_virtual_mouse
from virtual_keyboard import run_virtual_keyboard

if __name__ == "__main__":
    print("Choose mode:\n1. Virtual Mouse\n2. Virtual Keyboard")
    choice = input("Enter choice: ")

    if choice == "1":
        run_virtual_mouse()
    elif choice == "2":
        run_virtual_keyboard()
    else:
        print("Invalid choice")
