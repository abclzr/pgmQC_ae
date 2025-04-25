import numpy as np

Hadamard = 1 / np.sqrt(2) * np.array([
    [1, 1],
    [1, -1]
])

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

I = np.matrix([[1,0],[0,1]])
X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])
Z = np.matrix([[1,0],[0,-1]])
Hadamard = np.matrix([[1, 1], [1, -1]])/np.sqrt(2)

def CPhase(gama=None):
    assert gama is not None
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, np.exp(gama*(0+1j))]]) # type: ignore

def U3(theta, phi, lam):
    return np.matrix([[np.cos(theta/2), - np.sin(theta/2) * np.exp(lam*(0+1j))],
                      [np.sin(theta/2) * np.exp(phi*(0+1j)), np.cos(theta/2) * np.exp((phi+lam)*(0+1j))]])

def RX(theta=None):
    assert theta is not None
    return np.matrix([[np.cos(theta/2), -1j*np.sin(theta/2)],
                      [-1j*np.sin(theta/2), np.cos(theta/2)]])

def RY(theta=None):
    assert theta is not None
    return np.matrix([[np.cos(theta/2), -np.sin(theta/2)],
                      [np.sin(theta/2), np.cos(theta/2)]])

def RZ(theta=None):
    assert theta is not None
    return np.matrix([[np.cos(theta/2)-1j*np.sin(theta/2), 0],
                      [0, np.cos(theta/2)+1j*np.sin(theta/2)]])

def RZZ(theta=None):
    assert theta is not None
    return np.matrix([[np.cos(theta/2)-1j*np.sin(theta/2), 0, 0, 0],
                      [0, np.cos(theta/2)+1j*np.sin(theta/2), 0, 0],
                      [0, 0, np.cos(theta/2)+1j*np.sin(theta/2), 0],
                      [0, 0, 0, np.cos(theta/2)-1j*np.sin(theta/2)]])