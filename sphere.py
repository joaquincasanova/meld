import numpy as np

def vect_magnitude(R):
    return np.sqrt(np.transpose(sum(np.transpose(R**2),0)))

def cart2sph(x,y,z):
    az = np.arctan2(y,x)
    el = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return az, el, r

def sph2cart(az,el,r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def vect_cart2sph( Ax,Ay,Az,az,el ):
    Aaz=np.ones(np.shape(Ax))
    Ael=np.ones(np.shape(Ax))
    Ar=np.ones(np.shape(Ax))
#converts vector in cartesian coordinates to vector in sperical
    if np.size(az)==1:
        az=az*np.ones(np.shape(Ax))
        el=el*np.ones(np.shape(Ax))
    
    for i in range(0,np.size(Ax)):
        A = [[-np.sin(az[i]),                        np.cos(az[i]),         0],
             [-np.sin(el[i])*np.cos(az[i]), -np.sin(el[i])*np.sin(az[i]),np.cos(el[i])],        
             [ np.cos(el[i])*np.cos(az[i]), np.cos(el[i])*np.sin(az[i]),np.sin(el[i])]]
        inn = np.hstack([Ax,Ay,Az])
        out=np.matmul(A,inn[i,:])
        Aaz[i]=out[0]
        Ael[i]=out[1]
        Ar[i]=out[2]
            
    return Aaz,Ael,Ar 

def vect_sph2cart(Aaz, Ael,Ar ,az,el ):
    Ax=np.ones(np.shape(Ar))
    Ay=np.ones(np.shape(Ar))
    Az=np.ones(np.shape(Ar))

    #converts vector in sperical coordinates to vector in cartesian
    if np.size(az)==1:
        az=az*np.ones(np.shape(Ax))
        el=el*np.ones(np.shape(Ax))

        for i in range(0,np.size(el)):
            A = [[-np.sin(az[i]),-np.sin(el[i])*np.cos(az[i]),np.cos(el[i])*np.cos(az[i])],
                 [ np.cos(az[i]),-np.sin(el[i])*np.sin(az[i]),np.cos(el[i])*np.sin(az[i])],
                 [          0,            np.cos(el[i]),           np.sin(el[i])]]
            inn = np.hstack([Aaz,Ael,Ar])
            out=np.matmul(inn[i,:])
            Ax[i]=out[0]
            Ay[i]=out[1]
            Az[i]=out[2]


    return Ax,Ay,Az
