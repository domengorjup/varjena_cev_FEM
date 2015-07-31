import numpy as np
from scipy.integrate import quad
from numpy.linalg import solve
import matplotlib
from matplotlib import pyplot as plt

np.set_printoptions(precision=5)

#Podatki: ---------------------------------------------------------------------
#Material
E_jeklo = 2.1e5     #[MPa]
nu_jeklo = 0.3
E_zvar = 2.05e5     #[MPa]
nu_zvar = 0.3
sigma_tec = 250     #[MPa] (ASTM A36 jeklo)
SF = 0.5            #[/] faktor varnosti
koef_zvara = 0.7    #[/] sigma_dop_zvar = sigma_dop * koef_zvara

#Geometrija
d_o = 1016          #[mm]           
t = 9.5             #[mm]
alpha_zvar = 30     #[°] - kot žleba zvara
R_koren = 10        #[mm] - širina špranje korena zvara

#Obremenitev
p = 1               #[MPa]
# -----------------------------------------------------------------------------

#Frormat prikaza: -------------------------------------------------------------
'''
simetrija =     1: prikaz celotne cevi (simetrija y)
                0: prikaz le polovice cevi 
                
prikazi:        0   ->  primerjalna napetost
                        
                1   ->  sigma_xx
                2   ->  sigma_yy
                3   ->  sigma_zz
                4   ->  sigma_xy
                        
                5   ->  epsilon_xx
                6   ->  epsilon_yy
                7   ->  epsilon_xy
(primer: prikazi = [0,1,2])

povecava =      <vrednost povačave prikaza pomikov>
'''
simetrija  = 1
povecava = 1
prikazi = [0]
# -----------------------------------------------------------------------------

#Uvoz (definicija) mreže: -----------------------------------------------------
mesh_import = 1                 #1: uvoz mreže,     2: definiraj lastno mrežo
nodes_file = "nodes.txt"
elements_file = "elements.txt"
# -----------------------------------------------------------------------------


#Preračun vhodnih podatkov ----------------------------------------------------
r_o = d_o/2
r_i = r_o - t
dz = 1              #[mm]

sig_dop = sigma_tec * SF
sig_dop_zvar = sig_dop * koef_zvara

# Uboz mreže ------------------------------------------------------------------
def readfile(text,elements=0):
    '''
    text:       ime datoteke s podatki o vozliščih / elementih
    elements:   1, ko uvazamo datoteko z elementi (privzeto 0)
    '''
    lines = []
    with open(text) as textfile:
        for line in textfile.readlines():
            line = line.translate({ord(','):None})  # odstrani ','
            line = line.rstrip().split()[1:]        # razdeli, odstrani 1. element
            if elements:
                lines.append([int(i)-1 for i in line])  #pretvori v int, odšteje 1
            else:
                lines.append([float(i) for i in line])  #pretvori v float
            
    return lines

if mesh_import: 
    print('Uvoz mreže...\n')
    nodes = readfile(nodes_file)
    elements = readfile(elements_file,1)
    

else:
    #Točke
    nodes = [[0.,0.],
             [0.,-r_i],             
             [r_i,-r_i],
             [r_i,0.],
             [r_i,r_i],
             [0.,r_i]]

    #Povezave točk - elementi:
    elements = [[0,1,3],
                [3,5,0],
                [1,2,3],
                [3,4,5]]


nodearray=np.array(nodes,dtype=float)            

#Mreža:
def build_mesh(nodearray, elements):
    mesh = []
    for el in elements:
        mesh.append([nodearray[el,:],el])
        #element seznama mesh: [xy,el], "el" je seznam nodov elementa

    return mesh

mesh = build_mesh(nodearray, elements)


print('Začetek analize...')

# ROBNI POGOJI --------------------------------------------------------
#Bistveni (fiksirani pomiki): -----------------------------------------
xfixed = []

for i in range(nodearray.shape[0]):
    if nodearray[i][1] == 0:
        zero_y = i
        break
    
yfixed = [zero_y]    #Eno vozlišče je treba fiksirati v y smeri 

#Fiksirani pomiki v x smeri za vse node na simetrijski osi:
for n in range(nodearray.shape[0]):
    if nodearray[n,0] == 0:
        xfixed.append(n)


xvalue=np.zeros(2*nodearray.shape[0])  #zaenkrat le, če so fiksirani pomiki 0
yvalue=np.zeros(2*nodearray.shape[0])


#Iskanje vozlišč na notranjem robu cevi
def r(node):
    return np.sqrt(node[0]**2+node[1]**2)

def phi(node):
    return np.arctan2(node[1],node[0])

notranji = []
for n in range(nodearray.shape[0]):
    if np.abs(r(nodearray[n,:])-r_i) <= r_i/10000:
        notranji.append(n)


# Območje zvara (oblika trapeza - desna premica) ---------------------
k_zvar = np.tan(np.pi/2 - alpha_zvar/2/180*np.pi)
y_zvar = np.sqrt(r_i**2-(R_koren/2)**2)                 
T1_zvar = [R_koren/2, y_zvar]
n_zvar = T1_zvar[1]-T1_zvar[0]*k_zvar


# -------------------------------------------------------------------

class FElement(object):
    ''' En končni element, s koordinatami vozlišč '''

    def __init__(self,mesh):
        self.xy = mesh[0]       #np.array 3x2, z x,y koordiantami vozlišč
        self.nodes = mesh[1]    #seznam vozlišč elementa 
        
        if self.is_weld():          #Če je element v območju zvara drug matrial
            self.E = E_zvar
            self.nu = nu_zvar
        else:
            self.E = E_jeklo    
            self.nu = nu_jeklo
        
        self.area()
        self.B()
        self.D()
        self.K()
        self.scatter()
        
        
        if self.is_inner():         #Če gre za notranji element cevi 
            self.f_element()        #izračunaj vozliščne sile
        else: 
            self.f_el = np.zeros(len(self.dofs))    #sicer same ničle 
                                  
            
    def area(self):
        x1,y1=self.xy[0]
        x2,y2=self.xy[1]
        x3,y3=self.xy[2]

        self.area=np.abs(1/2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))


    def B(self):
        A = self.area
        
        def beta(i):
            return self.xy[i%3][1]-self.xy[(i+1)%3][1]
        def gamma(i):
            return self.xy[(i+1)%3][0]-self.xy[i%3][0]
      
        BB = np.array([[beta(1),0,beta(2),0,beta(3),0],
                       [0,gamma(1),0,gamma(2),0,gamma(3)],
                       [gamma(1),beta(1),gamma(2),beta(2),gamma(3),beta(3)]],
                      dtype=float)

        self.B = 1/(2*A)*BB


    # Ravninsko deformacijsko stanje:
    def D(self):
        DD = np.array([[1-self.nu,self.nu,0],
                       [self.nu, 1-self.nu,0],
                       [0,0,(1-2*self.nu)/2]], dtype=float)

        self.D = self.E/((1+self.nu)*(1-2*self.nu)) * DD


    #Togostna matrika elementa:
    def K(self):
        self.K = np.dot(np.dot(np.transpose(self.B),self.D),self.B) * self.area * dz


    #Vektor vozliščnih sil elementa
    def f_element(self):
        x1,y1 = self.notranji[0]
        x2,y2 = self.notranji[1]
        a = np.sqrt(np.abs(x1-x2)**2 + np.abs(y1-y2)**2)
        phi_F = np.pi/2 - np.arctan(np.abs(y1-y2)/np.abs(x1-x2)) # kot lokalne voz. sile s horizontalo
        F_voz = p * a/2     # velikost vozliščne sile v lokalne k. sistemu
        
        f_el = np.zeros(len(self.dofs))      # pripravim vektor vozliščnih sil elementa
        for i in self.skupni:
            xsign = np.sign(np.cos(phi(self.xy[i])))    # predznak x komponente
            ysign = np.sign(np.sin(phi(self.xy[0])))    # predznak y komponente
            f_el[2*i]  = F_voz*np.cos(phi_F) * xsign
            f_el[2*i+1] = F_voz*np.sin(phi_F) * ysign
        
        self.f_el = f_el    # vektor vozliščnih sil v lokalnem k.s. elementa
        
        
    #Razporeditev elementa v globalno togostno matriko:
    def scatter(self):
        dofs = []
        for n in self.nodes:
            dofs.extend((2*n,2*n+1))
        self.dofs = dofs
    
    
    #Ali je element na notranjem robu cevi:
    def is_inner(self):
        skupni = []
        for i in range(len(self.nodes)):
            if self.nodes[i] in notranji:
                skupni.append(i)
        
        if len(skupni)==2: 
            self.notranji = [self.xy[i] for i in skupni]  #Dve vozlišči na notranjem robu cevi
            self.skupni = skupni            
            return(1)


    #Ali je element na območju zvara: 
    def is_weld(self):
        self.centroid()
        if self.tez[1] >= y_zvar and self.tez[0] <= (self.tez[1]-n_zvar)/k_zvar:
            return 1
        
        else: return 0
            
    #Težišče trikotnega elementa:
    def centroid(self):
        self.tez = [np.sum(self.xy[:,0])/3, np.sum(self.xy[:,1])/3]



#Vsi elementi v mreži: --------------------------------------------------------
FE=[]               # seznam vseh končnih elementov v mreži
for m in mesh:
    FE.append(FElement(m))


#Globalna togostna matrika: ---------------------------------------------------
def build_K(FE,K_size) :   

    Kg = np.zeros([K_size,K_size])
    
    for el in FE:
        for i in range(len(el.dofs)):
            for j in range(len(el.dofs)):
                Kg[el.dofs[i],el.dofs[j]] += el.K[i,j] 

    return Kg
    
K_size = len(nodes)*2

# -----------------------------------------------------------------------------
Kg = build_K(FE,K_size)
# -----------------------------------------------------------------------------

#Vektor vozliščnih sil --------------------------------------------------------
def build_f_tlak(FE,size):
    
    fg = np.zeros(size)
    
    for el in FE:
        for i in range(len(el.dofs)):
            fg[el.dofs[i]] += el.f_el[i]
            
    return fg

# -----------------------------------------------------------------------------
f = build_f_tlak(FE,K_size)
# -----------------------------------------------------------------------------


#Upoštevanje bistvenih robnih pogojev (preoblikovanje enačbe):
Kn = np.copy(Kg)
fn = np.copy(f)

for i in xfixed:
    Kn[2*i,:]=0
    Kn[:,2*i]=0
    Kn[2*i,2*i]=1

    fn[:]-=Kg[:,2*i]*xvalue[i]
    fn[i*2]=xvalue[i]

for i in yfixed:
    Kn[2*i+1,:]=0
    Kn[:,2*i+1]=0
    Kn[2*i+1,2*i+1]=1

    fn[:]-=Kg[:,2*i+1]*yvalue[i]
    fn[i*2+1]=yvalue[i]


#Rešitev sistema: -------------------------------------------------------------
U = solve(Kn,fn)
F = np.dot(Kg,U)

print('Konec analize.\n')


#Postprocesiranje: ------------------------------------------------------------
print('Postprocesiranje...\n')

#Nove koordinate vozlišč: -----------------------------------------------------
U_nodes = U.reshape(nodearray.shape)
new_nodes = nodearray + U_nodes 

#Deformacije in napetosti: ----------------------------------------------------
eps = []
for element in FE:
    eps.append(np.dot(element.B, U[element.dofs]))

sig = []
for i in range(len(FE)):
    sig.append(np.dot(FE[i].D, eps[i]))
    
for i in range(len(sig)):
    sig[i] = np.append(sig[i], FE[i].nu*(sig[i][0]+sig[i][1]))  #sigma_zz

deformacije = np.array(eps)
napetosti = np.array(sig)

#Primerjalne napetosti (Von Mieses):
sig_VM = np.array([np.sqrt(s[0]**2+s[1]**2+s[3]**2-s[0]*s[1]-s[1]*s[3]-s[0]*s[3]+3*s[2]**2) for s in sig], dtype=float)


#Elementi zvara: --------------------------------------------------------------
zvar = []
for i in range(len(FE)):
    if FE[i].is_weld():
        zvar.append(i)
        
FE_zvar = [FE[i] for i in zvar]
sig_zvar = sig_VM[zvar]
U_zvar = [U_nodes[FE[i].nodes] for i in zvar]


#Prikaz: ----------------------------------------------------------------------
prikaz = 1

za_prikaz = [{'data': sig_VM, 'naslov': 'Primerjalna napetost'},
             {'data': napetosti[:,0], 'naslov': r'$\sigma_{xx}$ [MPa]'},
             {'data': napetosti[:,1], 'naslov': r'$\sigma_{yy}$ [MPa]'},
             {'data': napetosti[:,3], 'naslov': r'$\sigma_{zz}$ [MPa]'},
             {'data': napetosti[:,2], 'naslov': r'$\sigma_{xy}$ [MPa]'},
             {'data': deformacije[:,0], 'naslov': r'$\varepsilon_{xx}$ [/]'},
             {'data': deformacije[:,1], 'naslov': r'$\varepsilon_{yy}$ [/]'},
             {'data': deformacije[:,2], 'naslov': r'$\varepsilon_{xy}$ [/]'}]
             

def plot_mesh(mesh,style,sym=0):
    for m in mesh:
        x = np.append(m[0][:,0], m[0][0,0])
        y = np.append(m[0][:,1], m[0][0,1])
        
        plt.plot(x,y,style)
        
        if sym:
            plt.plot(-x,y,style)
        
        
def plot_fill(value_array,title,sym=0):
    x = nodearray[:,0]
    y = nodearray[:,1]
    triangles = np.array(elements)
    
    if sym:
        x = np.append(x,-x)
        y = np.append(y,y)
        triangles = np.vstack((triangles, triangles+np.amax(triangles)+1))
        value_array = np.append(value_array,value_array) 
    
    plt.figure()
    plt.title(title)
    plt.axes().set_aspect('equal')        
    plt.tripcolor(x,y,triangles,value_array, edgecolors='k',cmap=plt.get_cmap('jet'))
    plt.colorbar()


def plot_weld(value_array, sym=0):
    x = nodearray[:,0]
    y = nodearray[:,1]
    triangles = np.array(elements)

    xmin, xmax = (0, 1.5*R_koren/2+t*np.sin(alpha_zvar/2/180*np.pi))
    ymin, ymax = (r_i-xmax/2, r_o+xmax/2)

    if sym:
        x = np.append(x,-x)
        y = np.append(y,y)
        triangles = np.vstack((triangles, triangles+np.amax(triangles)+1))
        value_array = np.append(value_array,value_array)

        xmin = -xmax

    plt.figure()
    plt.title('Primerjalna napetost v zvaru [MPa]')
    plt.axes().set_aspect('equal')

    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])

    odmik = (xmax-xmin)/50              #odmik besedila od roba
    axes.text(xmin+odmik, ymin+odmik,
              "Največja primerjalna napetost:  {:.3f} MPa\nDopustna napetost:  {:.3f} MPa".format(np.amax(sig_zvar), sig_dop_zvar))
    
    plt.tripcolor(x,y,triangles,value_array, edgecolors='k',cmap=plt.get_cmap('jet'), vmin=0, vmax=sig_dop_zvar)
    plt.colorbar()
    
    for i in range(len(sig_zvar)):
        xy_tez = FE_zvar[i].tez
        axes.text(xy_tez[0], xy_tez[1], "{:.2f}".format(sig_zvar[i]), ha='center')
        if sym:        
            axes.text(-xy_tez[0], xy_tez[1], "{:.2f}".format(sig_zvar[i]), ha='center')
        


def printU(element):
        print(U_nodes[FE[i].nodes])


if prikaz:
    print('Generiranje prikaza...')

    #Elementi
    plt.figure()
    plt.grid()
    plt.axes().set_aspect('equal')
    plt.title('Deformirana oblika (faktor povečave: {:.1f})'.format(povecava))
    
    plot_mesh(mesh, '--k', sym=simetrija)
    deformed = build_mesh(nodearray + U_nodes*povecava, elements)
    plot_mesh(deformed, '-b', sym=simetrija)
    plot_mesh([deformed[i] for i in zvar],'-r', sym=simetrija)     #elementi zvara

    #Napetosti, specifične deformacije:
    for dataset in [za_prikaz[i] for i in prikazi]:
        plot_fill(dataset['data'], dataset['naslov'], sym=simetrija)

    #Približani elementi zvara
    plot_weld(sig_VM,sym=simetrija)


    plt.show()
