# ─────────────────────────────────────────────────────────────────────────────
# User interface Test
#
# This test does not involve AI.  It simply tests the User Interface in app.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("Welcome to the Assistant!\n")
print("-- Type 'end' to exit the assistant. --\n")

while True:
    name = input("\nEnter your name: ")
    if name.lower() == "end":
        print("Goodbye!")
        break
    
    print("Name:", name)