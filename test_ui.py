
# ─────────────────────────────────────────────────────────────────────────────
# User interface Test
#
# This test does not involve AI.  It simply tests the User Interface in app.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("Welcome to UI Test!\n")
print("-- Type 'end' to exit the assistant. --")

while True:
    user_input = input("\nEnter some text: ")
    if user_input.lower() == "end":
        print("Goodbye!")
        break
    
    print("User input:", user_input)