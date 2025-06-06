from time import sleep  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def ouvrir(data):
    my_file = open(data, "r")
    data = my_file.read()
    my_file.close()
    
    data = data.split("\n")
    data = [[int(data[i][j]) for j in range(len(data[0]))] for i in range(len(data))]
    return(data)

def evolution(data):     
        
    #créer matrice suivante
    next_data = [[0 for j in range(len(data[0]))] for i in range(len(data))]
    
    #initialiser les bords
    for i in range(len(data)):
        next_data[0][i] = data[0][i]
        next_data[len(data)-1][i] = data[len(data)-1][i]
        next_data[i][0] = data[i][0]
        next_data[i][len(data)-1] = data[i][len(data)-1]
        
    #fais évoluer chaque cellule hors bord
    for i in range(1,len(data)-1):
        for j in range(1,len(data)-1):
            
            #compte les voisins
            neighbours = data[i-1][j-1]+data[i-1][j]+data[i-1][j+1] \
             + data[i][j-1]+data[i][j+1] \
             + data[i+1][j-1]+data[i+1][j]+data[i+1][j+1]
            
            #détermine l'état de la cellule            
            if data[i][j]==0 and neighbours == 3:
                next_data[i][j] = 1
                
                
            if data[i][j]==1:
                if neighbours in [2,3]:
                    next_data[i][j] = 1
                    
                else:
                    next_data[i][j] = 0
                    
    return(next_data)

def afficher_terminal(data):
    
    print("-" * (len(data)+2))
    
    for i in range(len(data)):
        clean = "|"
        for j in range(len(data)):
            if data[i][j]==1:
                clean = clean + "¤"
            else:
                clean = clean + " "
        print(clean+"|")
    
    print("-" * (len(data)+2))
    
def enregistrer(data):
    f = open('output',"w")
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            f.write(str(data[i][j]))
        if i != len(data)-1:
            f.write("\n")
    
    f.close()

def afficher(data):
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gray')
    ax.axis('off')
    plt.show()
    

# lecture du fichier input
size = int(input("Enter a size for a random grid or 0 for the 'input' map\n"))

if size == 0:
    data = ouvrir("input")
else:
    data = [[ np.random.randint(0,2) for j in range (size)] for i in range (size)]

start = data

# affichage de 110 évolutions (terminal et matplotlib)
#for i in range(110):
#    data=evolution(data)
#    afficher(data)
#    afficher_terminal(data)
#    print(i+1)
#    sleep(0.1)

# affichage simultané de input et output
#fig, ax = plt.subplots(1,2)
#ax[0].imshow(start, cmap = 'gray')
#ax[0].set_title('input')
#ax[0].axis('off')
#ax[1].imshow(data, cmap = 'gray')
#ax[1].set_title('output')
#ax[1].axis('off')
#plt.show()

# affichage avec animation

def updater(i, img):
    global data
    data = evolution(data)
    img.set_data(data)
    return img

fig, ax = plt.subplots()
ax.axis('off')
img = ax.imshow(data, cmap='gray')
    

ani = animation.FuncAnimation(fig , updater , fargs = (img,) ,\
frames =100 , repeat = False , interval =200)
ani.save('life.gif')    
plt.show ()

# écriture de la dernière évolution dans output
enregistrer(data)






