from numpy import random
from getpass import getuser
from time import sleep

print('Bonjour', getuser())

MIN_NUMBER = 100
MAX_NUMBER = 200

def game():
    ''' The game in itself    
    '''
    
    # The random number to guess
    r = random.randint(MIN_NUMBER,MAX_NUMBER)
    found = False
    
    # (Eternal) loop
    #while not found:
    while True: #boucle infinie
        
        entry = input("\nEnter a number between "+str(MIN_NUMBER)+" and "+str(MAX_NUMBER)+": ")
        while not entry.isdigit():
            print("please input a number")
            entry = input("\nEnter a number between "+str(MIN_NUMBER)+" and "+str(MAX_NUMBER)+": ")

        entry = int(entry)
        # Condition on what to do
        if entry !=0 and (entry < MIN_NUMBER or entry > MAX_NUMBER):
            print("you are out of bounds, take 10 secs to get it !")
            sleep(10)                    
        elif entry == r or entry ==0 or getuser()=='didier': # 0 pour gagner (cheat !)
            print("\n\nGood job, it was "+str(r)+"!!!")
            found=True
        elif entry>r:
            print("You're too high!\nHINTS: substract more than",(entry-r)//10*10)
        else:
            print("A bit more?\nHINTS: Add more than",(r-entry)//10*10)
    	
 
# Start the game only if you wish
#playerwish = input("Hello, want to play a game? ")
#if playerwish in ["yes", "y"] :
#	game()

game()	


