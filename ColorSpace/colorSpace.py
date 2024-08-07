import numpy as np

def RgbToHsv(RGB):
    RgbNormalized = RGB / 255.0       #In this part what I do is normalize the RGB channel and divide it              
    
    R = RgbNormalized[:, :, 0]                            
    G = RgbNormalized[:, :, 1]
    B = RgbNormalized[:, :, 2]
    
    vMax = np.max(RgbNormalized, axis=2)                
    vMin = np.min(RgbNormalized, axis=2)
    C = vMax - vMin                                      
    
    rMax = np.logical_and(R == vMax, C > 0)     #Hue calculation depending on the value range 
    gMax = np.logical_and(G == vMax, C > 0)
    bMax = np.logical_and(B == vMax, C > 0)

    H = np.zeros_like(vMax)                               
    
    hR = (((G[rMax] - B[rMax]) / C[rMax]) % 6)*60
    hG = (((B[gMax] - R[gMax]) / C[gMax]) + 2)*60
    hB = (((R[bMax] - G[bMax]) / C[bMax]) + 4)*60
    
    H[rMax] = hR
    H[gMax] = hG
    H[bMax] = hB
    
    V = vMax                                     #Value Calculation 
    
    S = np.zeros_like(vMax)                      #Saturation Calculation
    S[V>0] = C[V>0] / V[V>0]
    
    return np.dstack((H, S, V))

def HsvToRgb(HSV):
    H = HSV[:, :, 0]                                                         # Split HSV Channel
    S = HSV[:, :, 1]
    V = HSV[:, :, 2]
    
    C = V * S                                                  
    
    hNormalize = H / 60.0                                                    # Normalize hue
    X  = C * (1 - np.abs(hNormalize % 2 - 1))                                # Compute value of 2nd largest color
    
    hRange1 = np.logical_and(0 <= hNormalize, hNormalize<= 1)                    
    hRange2  = np.logical_and(1 <  hNormalize, hNormalize<= 2)
    hRange3  = np.logical_and(2 <  hNormalize, hNormalize<= 3)
    hRange4  = np.logical_and(3 <  hNormalize, hNormalize<= 4)
    hRange5  = np.logical_and(4 <  hNormalize, hNormalize<= 5)
    hRange6  = np.logical_and(5 <  hNormalize, hNormalize<= 6)
    
    rgbColorValues = np.zeros_like(HSV)                                       #Color values
    Z = np.zeros_like(H)
    
    rgbColorValues[hRange1] = np.dstack((C[hRange1], X[hRange1], Z[hRange1]))  
    rgbColorValues[hRange2] = np.dstack((X[hRange2], C[hRange2], Z[hRange2]))
    rgbColorValues[hRange3] = np.dstack((Z[hRange3], C[hRange3], X[hRange3]))
    rgbColorValues[hRange4] = np.dstack((Z[hRange4], X[hRange4], C[hRange4]))
    rgbColorValues[hRange5] = np.dstack((X[hRange5], Z[hRange5], C[hRange5]))
    rgbColorValues[hRange6] = np.dstack((C[hRange6], Z[hRange6], X[hRange6]))
    
    m = V - C
    RGB = rgbColorValues + np.dstack((m, m, m))                        
    
    return RGB

def RgbToYcbcr(RGB):
    xform = np.array([[.299, .587, .114], [-.169, -.331, .5], [.5, -.419, -.081]])
    ycbcr = RGB.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def YcbcrTorgb(YCBCR):
    xform = np.array([[1, 0, 1.403], [1, -0.344, -.714], [1, 1.773, 0]])
    rgb = YCBCR.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)