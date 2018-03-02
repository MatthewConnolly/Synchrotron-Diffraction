# %% 1
# Package imports
import matplotlib.pyplot as plt
import math
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import scipy
from numpy import exp
import itertools
import pandas as pd
from matplotlib.lines import Line2D
import pdb

# Plotting helpers
colors = ['k','b','g','r','m']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

#Define a crack class
class Crack:

    Youngs = 226000 #MPa
    nu = 0.287 #Unitless
    YieldStrength = 696.0 #MPa
    reflections = ["110", "200", "220", "310", "222"]
    datadir = "/home/matt/Q/Experiments/1-ID Summer 2017/Data/1-ID Summer 2017 Tomo Files/connolly_jun17/matlab/Analysis/"
    dislocationdir = "/home/matt/Q/Experiments/1-ID Summer 2017/Acta Material Paper/Images/DislocationData/"
    
    def __init__(self, env, load, B, W, d0, length, xpos, ypos, strainfile, disfile):

        self.env = env
        self.load = load
        self.length = length
        self.xpos = xpos
        self.ypos = ypos
        self.B = B
        self.W = W
        self.d0 = d0

        self.strainfile = self.datadir+strainfile
        self.data = pd.read_csv(self.strainfile, delim_whitespace=True)
        self.X, self.Y, self.epsXX, self.epsYY, self.epsXY, \
            self.depsXX,  self.depsYY,  self.depsXY = self.set_Strain()
        self.planeStrainSigmaXX, self.planeStrainSigmaYY, self.planeStrainSigmaXY, self.planeStrainSigmaZZ, \
            self.planeStraindSigmaXX, self.planeStraindSigmaYY, self.planeStraindSigmaXY, self.planeStraindSigmaZZ = self.set_Stress_PlaneStrain()
        self.theta, self.PrincipalStrain, self.PrincipalStrainXX, self.PrincipalStrainYY, \
            self.dtheta, self.dPrincipalStrainXX, self.dPrincipalStrainYY = self.set_Principal_Strain()

        self.PrincipalStressXX, self.PrincipalStressYY, self.PrincipalStressZZ, \
            self.dPrincipalStressXX, self.dPrincipalStressYY, self.dPrincipalStressZZ = self.set_Principal_Stress()
        self.HydrostaticStress, \
            self.dHydrostaticStress = self.set_Hydrostatic_Stress()
        self.vonMisesStress, \
            self.dvonMisesStress = self.set_vonMises_Stress()
        self.eta, \
            self.deta = self.set_triaxiality()
        
        self.Kscatt, self.KFWHM, self.KFWHMCorr, self.tthFWHMCorr, self.dcosFWHMCorr = self.set_FWHM()
        
        self.discfile = self.dislocationdir+disfile
        self.discdata = pd.read_csv(self.discfile)
        # self.discX, self.discDens = self.set_discDensity()

        self.KASTM = self.calc_K_ASTM()
        self.PZsize_Analytical = self.calc_PlasticZoneSize_Analytical()

        self.discDens = self.calc_discDensity()

        self.KStressFit, self.KStressFitError, self.KStressFitLineX, self.KStressFitLineY = self.calc_K_StressFit()

    def set_Strain(self):
        
        XFromCrackTip = np.asmatrix(self.data["sampX"]) - self.xpos
        YFromCrackTip = np.asmatrix(self.data["sampY"]) - self.ypos

        if(self.env == "Hydrogen"):
            dXX = np.asmatrix(self.data["dH1"])-0.0011*np.asmatrix(self.data["sampX"])
            dYY = np.asmatrix(self.data["dV1"])-0.0011*np.asmatrix(self.data["sampX"])
            dXY = np.asmatrix(self.data["dXY1"])-0.0011*np.asmatrix(self.data["sampX"])
            
        else:
            dXX = np.asmatrix(self.data["dH1"])+0.0011*np.asmatrix(self.data["sampX"])
            dYY = np.asmatrix(self.data["dV1"])+0.0011*np.asmatrix(self.data["sampX"])
            dXY = np.asmatrix(self.data["dXY1"])+0.0011*np.asmatrix(self.data["sampX"])
            
        ddXX = np.asmatrix(self.data["ddH1"])
        ddYY = np.asmatrix(self.data["ddV1"])
        ddXY = np.asmatrix(self.data["ddXY1"]) / 4.0 # Factor of four was not incorporated in matlab file, so doing so here.
        
        epsXX = 10**6 * (dXX - self.d0)/self.d0
        epsYY = 10**6 * (dYY - self.d0)/self.d0
        epsXY = 10**6 * (dXY - self.d0)/self.d0
        depsXX = 10**6 * ddXX/self.d0
        depsYY = 10**6 * ddYY/self.d0
        depsXY = 10**6 * ddXY/self.d0

        return (XFromCrackTip, YFromCrackTip, epsXX, epsYY, epsXY, depsXX, depsYY, depsXY)
        
    def set_Stress_PlaneStrain(self):
    
        trueepsXX = 10**-6 * np.asarray(self.epsXX)
        trueepsYY = 10**-6 * np.asarray(self.epsYY)
        trueepsXY = 10**-6 * np.asarray(self.epsXY)
        truedepsXX = 10**-6 * np.asarray(self.depsXX)
        truedepsYY = 10**-6 * np.asarray(self.depsYY)
        truedepsXY = 10**-6 * np.asarray(self.depsXY)
        
        sigXX = (self.Youngs/((1-2*self.nu)*(1+self.nu)))*((1-self.nu)*trueepsXX+self.nu*trueepsYY)
        sigYY = (self.Youngs/((1-2*self.nu)*(1+self.nu)))*((1-self.nu)*trueepsYY+self.nu*trueepsXX)
        sigXY = (self.Youngs*trueepsXY)/(2*(1+self.nu))
        sigZZ = ((self.Youngs*self.nu)/((1-2*self.nu)*(1+self.nu)))*(trueepsYY+trueepsXX)
        
        dsigXX = np.sqrt(((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(1-self.nu))**2*(truedepsXX)**2 \
                    + ((self.Youngs/((1-2*self.nu)*(1+self.nu)))*self.nu)**2*np.power(truedepsYY,2))
        dsigYY = np.sqrt(((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(1-self.nu)*truedepsYY)**2 \
                    + ((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(self.nu*truedepsXX))**2)
        dsigXY = np.sqrt(((self.Youngs*truedepsXY)/(2*(1+self.nu)))**2)
        dsigZZ = np.sqrt((((self.Youngs*self.nu)/((1-2*self.nu)*(1+self.nu)))*(truedepsYY))**2 + (((self.Youngs*self.nu)/((1-2*self.nu)*(1+self.nu)))*(truedepsXX))**2)
        
        return (sigXX, sigYY, sigXY, sigZZ, dsigXX, dsigYY, dsigXY, dsigZZ)
        
    def set_Principal_Strain(self):
    
        Rmat = np.asmatrix([[1,0,0],[0,1,0],[0,0,2]])
        epsmat = np.array([self.epsXX, self.epsYY, self.epsXY])
        def Amatfunc(theta):
            c = math.cos(theta)
            s = math.sin(theta)
            return np.matrix([[c**2, s**2, 2*s*c],[s**2, c**2, -2*s*c],[-s*c, s*c, c**2 - s**2]])

        def RAR(theta):
            return np.matmul(Rmat,np.matmul(Amatfunc(theta),Rmat.I))
            
        theta = 0.5*np.arctan2(np.array(self.epsXY),(np.array(self.epsXX)-np.array(self.epsYY)))

        dthetadepsXY = np.asarray(0.5*(1/np.asarray(self.epsXX +  self.epsYY + np.power(self.epsXY,2)/np.asarray(self.epsXX + self.epsYY)))) #Error currently here.
        dthetadepsXX = np.asarray(-0.5*(np.asarray(self.epsXY)/(np.power(self.epsXY,2) +  np.power(self.epsXX,2) + np.power(self.epsYY,2) + 2*np.asarray(self.epsXX)*np.asarray(self.epsYY))))
        dthetadepsYY = np.asarray(dthetadepsXX)
        dtheta = np.sqrt( np.power(dthetadepsXY,2) * np.power(np.asarray(self.depsXY),2) + np.power(dthetadepsXX,2) * np.power(np.asarray(self.depsXX),2) + np.power(dthetadepsYY,2) * np.power(np.asarray(self.depsYY),2) )

        depsprimeXXdtheta = 4 * np.asarray(self.epsXY)*np.cos(2*theta) + np.asarray((self.epsYY - self.epsXX))*np.sin(2*theta)
        depsprimeXXdepsXX = np.power(np.cos(theta),2)
        depsprimeXXdepsXY = 4 * np.cos(theta) * np.sin(theta)
        depsprimeXXdepsYY = np.power(np.sin(theta),2)

        depsprimeYYdtheta = np.asarray(self.epsYY) * np.power(np.cos(theta),2) - 4 * np.asarray(self.epsXY)*np.cos(theta)*np.sin(theta) + np.asarray(self.epsXX)*np.power(np.sin(theta),2)
        depsprimeYYdepsXX = np.power(np.sin(theta),2)
        depsprimeYYdepsXY = -4 * np.cos(theta) * np.sin(theta)
        depsprimeYYdepsYY = np.power(np.cos(theta),2)
        
        # depsprimeXX = np.sqrt(np.power(depsprimeXXdtheta,2) * np.power(dtheta,2) + np.power(depsprimeXXdepsXX,2) * np.power(np.asarray(self.depsXX), 2) \
                              # + np.power(depsprimeXXdepsXY,2) * np.power(np.asarray(self.depsXY),2) + np.power(depsprimeXXdepsYY,2) * np.power(np.asarray(self.depsYY),2))
        # depsprimeYY = np.sqrt(np.power(depsprimeYYdtheta,2) * np.power(dtheta,2) + np.power(depsprimeYYdepsXX,2) * np.power(np.asarray(self.depsXX), 2) \
                      # + np.power(depsprimeYYdepsXY,2) * np.power(np.asarray(self.depsXY),2) + np.power(depsprimeYYdepsYY,2) * np.power(np.asarray(self.depsYY),2))    

        depsprimeXX = np.sqrt(np.power(depsprimeXXdepsXY,2) * np.power(np.asarray(self.depsXY),2) + np.power(depsprimeXXdepsXX,2) * np.power(np.asarray(self.depsXX), 2)\
                                + np.power(depsprimeXXdtheta,2) * np.power(dtheta,2) + np.power(depsprimeXXdepsYY,2) * np.power(np.asarray(self.depsYY),2))
        depsprimeYY = np.sqrt(np.power(depsprimeYYdepsXY,2) * np.power(np.asarray(self.depsXY),2) + np.power(depsprimeYYdepsXX,2) * np.power(np.asarray(self.depsXX), 2)\
                                + np.power(depsprimeYYdtheta,2) * np.power(dtheta,2) + np.power(depsprimeYYdepsYY,2) * np.power(np.asarray(self.depsYY),2))                          
        
        # print depsprimeYY
        
        princStrain = []
        PrincipalStrainXX = []
        PrincipalStrainYY = []
        for i in range(len(np.asarray(self.epsXX)[0])):
            princStrain.append(RAR(np.asarray(theta)[0][i]).dot(np.array([np.array(self.epsXX)[0][i], np.array(self.epsYY)[0][i], np.array(self.epsXY)[0][i]])))
            PrincipalStrainXX.append(np.asarray(princStrain)[i][0][0])
            PrincipalStrainYY.append(np.asarray(princStrain)[i][0][1])
        return theta, princStrain, PrincipalStrainXX, PrincipalStrainYY, dtheta, depsprimeXX, depsprimeYY

    def set_Principal_Stress(self):

        princStressXX = []
        princStressYY = []
        princStressZZ = []
        dstrainXX = (10**-6) * np.asarray(self.dPrincipalStrainXX)
        dstrainYY = (10**-6) * np.asarray(self.dPrincipalStrainYY)
        
        dsigmaprimeXXdepsprimeXX = (self.Youngs*(1-self.nu))/((1-2*self.nu)*(1+self.nu))
        dsigmaprimeXXdepsprimeYY = self.Youngs*self.nu/((1-2*self.nu)*(1+self.nu))
        dsigmaprimeYYdepsprimeXX = dsigmaprimeXXdepsprimeYY
        dsigmaprimeYYdepsprimeYY = dsigmaprimeXXdepsprimeXX
        dsigmaprimeZZdepsprimeXX = self.Youngs*self.nu/((1-2*self.nu)*(1+self.nu))
        dsigmaprimeZZdepsprimeYY = dsigmaprimeZZdepsprimeXX
        
        dsigmaprimeXX = np.sqrt(np.power(dsigmaprimeXXdepsprimeXX,2) * np.power(dstrainXX,2) + np.power(dsigmaprimeXXdepsprimeYY,2)*np.power(dstrainYY,2))
        dsigmaprimeYY = np.sqrt(np.power(dsigmaprimeYYdepsprimeXX,2) * np.power(dstrainXX,2) + np.power(dsigmaprimeYYdepsprimeYY,2)*np.power(dstrainYY,2))
        dsigmaprimeZZ = np.sqrt(np.power(dsigmaprimeZZdepsprimeXX,2) * np.power(dstrainXX,2) + np.power(dsigmaprimeZZdepsprimeYY,2)*np.power(dstrainYY,2))
        
        for i in range(len(np.asarray(self.PrincipalStrain))):
            strainXX = (10**-6) * np.asarray(self.PrincipalStrain)[i][0][0]
            strainYY = (10**-6) * np.asarray(self.PrincipalStrain)[i][0][1]

            princStressXX.append((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(strainXX*(1-self.nu) + strainYY*self.nu))
            princStressYY.append((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(strainXX*(self.nu) + strainYY*(1-self.nu)))
            princStressZZ.append((self.Youngs/((1+self.nu))) * (self.nu/(1-2*self.nu)) * (strainXX + strainYY))

        return (princStressXX, princStressYY, princStressZZ, dsigmaprimeXX, dsigmaprimeYY, dsigmaprimeZZ)

    def set_Hydrostatic_Stress(self):
        sigmaH = (1.0/3.0)*(np.asarray(self.PrincipalStressXX) + np.asarray(self.PrincipalStressYY) + np.asarray(self.PrincipalStressZZ))
        dsigmaH = np.sqrt(np.power((1.0/3.0)*self.dPrincipalStressXX,2) + np.power((1.0/3.0)*self.dPrincipalStressYY,2) + np.power((1.0/3.0)*self.dPrincipalStressZZ,2))

        return sigmaH, dsigmaH
              
    def set_vonMises_Stress(self):
        
        vonMisesStress = []
        dsigmaVM = []
        
        for i in range(len(np.asarray(self.PrincipalStressXX))):
            s1 = self.PrincipalStressXX[i]
            s2 = self.PrincipalStressYY[i]
            s3 = self.PrincipalStressZZ[i]
            ds1 = self.dPrincipalStressXX.tolist()[0][i]
            ds2 = self.dPrincipalStressYY.tolist()[0][i]
            ds3 = self.dPrincipalStressZZ.tolist()[0][i]
            
            vonMisesStress.append(np.sqrt(0.5*((s1-s2)**2 + (s2-s3)**2 +(s1-s3)*3)))

            dsigmaVMdsigmaprimeXX = 0.5*(2*s1-s2-s3)/np.sqrt(s1**2 + s2**2 + s3**2 - s2*s3 - s1*s2 - s1*s3)
            dsigmaVMdsigmaprimeYY = - 0.5*(s1-2*s2+s3)/np.sqrt(s1**2 + s2**2 + s3**2 - s2*s3 - s1*s2 - s1*s3)
            dsigmaVMdsigmaprimeZZ = - 0.5*(s1+s2-2*s3)/np.sqrt(s1**2 + s2**2 + s3**2 - s2*s3 - s1*s2 - s1*s3)
            
            dsigmaVM.append(np.sqrt((dsigmaVMdsigmaprimeXX*ds1)**2 + (dsigmaVMdsigmaprimeYY*ds2)**2 + (dsigmaVMdsigmaprimeZZ*ds3)**2))

        return vonMisesStress, dsigmaVM

    def set_triaxiality(self):
    
        eta = []
        deta = []
        
        for i in range(len(np.asarray(self.vonMisesStress))):
        
            detadsigmaH = 1.0/self.vonMisesStress[i]
            detadsigmaVM = -self.HydrostaticStress[i]/np.power(self.vonMisesStress[i],2)
            
            deta.append(np.sqrt(np.power(detadsigmaH*self.dHydrostaticStress[0][i],2)))
            eta.append(self.HydrostaticStress[i]/self.vonMisesStress[i])
            
        return eta, deta
        
    def set_discDensity(self):
        
        XFromCrackTip = np.asmatrix(self.discdata["Xpos"])
        DislocationDensity = np.asmatrix(self.discdata["DiscDens"])
        return (XFromCrackTip, DislocationDensity)

    def set_FWHM(self):
        
        lam = 0.015498
        TiltCorrH = [0.00111, 0.00078, 0.00058, 0.00045, 0.00045]
        TiltCorrA = [-0.00103, -0.00076, -0.00052, -0.00047, -0.00041]
        dHList = ["dH1", "dH2", "dH3", "dH4", "dH5"]
        plusList = ["dHplus1", "dHplus2", "dHplus3", "dHplus4", "dHplus5"]
        minusList = ["dHminus1", "dHminus2", "dHminus3", "dHminus4", "dHminus5"]
        
        InstRes = 0.03439
        
        Kscatt = []
        KFWHM = []
        KFWHMCorr = []
        tthFWHMCorr = []
        dcosFWHMCorr = []
        
        if(self.env == "Hydrogen"):    
            for i, tc in enumerate(TiltCorrH):
                Kscatttemp = np.asarray(10/(np.asarray(self.data[dHList[i]]) + tc*np.asarray(self.data["sampX"])))
                KFWHMtemp = 10*np.abs((1/(np.asarray(self.data[plusList[i]]) + tc*np.asarray(self.data["sampX"]))) - (1/(np.asarray(self.data[minusList[i]]) + tc*np.asarray(self.data["sampX"]))))

                temp = [kt**2 - 0.03439**2 if kt**2 - 0.03439**2 > 0.0 else 0.0 for kt in KFWHMtemp]
                tthFWHMtemp = [2*np.arcsin((lam/2)*(Kscatttemp[i] + np.sqrt(t)/2)) - 2*np.arcsin((lam/2)*(Kscatttemp[i] - np.sqrt(t)/2)) for i, t in enumerate(temp)]
                dcosFWHMtemp = [tthFWHMtemp[i] * np.cos(np.arcsin(Kscatttemp[i]*lam/2)) / lam for i,t in enumerate(temp)]
                
                Kscatt.append(Kscatttemp)
                KFWHM.append(KFWHMtemp)
                KFWHMCorr.append(np.sqrt(temp))
                tthFWHMCorr.append(tthFWHMtemp)
                dcosFWHMCorr.append(dcosFWHMtemp)
               
        else:
            for i, tc in enumerate(TiltCorrA):        
                Kscatttemp = np.asarray(10/(np.asarray(self.data[dHList[i]]) + tc*np.asarray(self.data["sampX"])))
                KFWHMtemp = 10*np.abs((1/(np.asarray(self.data[plusList[i]]) + tc*np.asarray(self.data["sampX"]))) - (1/(np.asarray(self.data[minusList[i]]) + tc*np.asarray(self.data["sampX"]))))

                temp = [kt**2 - 0.03439**2 if kt**2 - 0.03439**2 > 0.0 else 0.0 for kt in KFWHMtemp]
                tthFWHMtemp = [2*np.arcsin((lam/2)*(Kscatttemp[i] + np.sqrt(t)/2)) - 2*np.arcsin((lam/2)*(Kscatttemp[i] - np.sqrt(t)/2)) for i, t in enumerate(temp)]
                dcosFWHMtemp = [tthFWHMtemp[i] * np.cos(np.arcsin(Kscatttemp[i]*lam/2)) / lam for i,t in enumerate(temp)]
                
                Kscatt.append(Kscatttemp)
                KFWHM.append(KFWHMtemp)
                KFWHMCorr.append(np.sqrt(temp))
                tthFWHMCorr.append(tthFWHMtemp)
                dcosFWHMCorr.append(dcosFWHMtemp)
                
        return np.asarray(Kscatt), np.asarray(KFWHM), np.asarray(KFWHMCorr), np.asarray(tthFWHMCorr), np.asarray(dcosFWHMCorr)

    def plot_discDensity(self):
            
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_DislocationDensity"
                # ax1.plot(Xfit,Yfit,color=color,linestyle='-',marker=marker,label=label)
        
        linescan = [[j[0], j[2]] for j in self.discDens if j[1] > -0.01 and j[1] < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]
        
        ax1.scatter(X, Y)
        ax1.set_title("Dislocation Density")    
        ax1.set_xlabel('Distance from crack tip (mm)')
        ax1.set_ylabel('Dislocation Density (mm$^{-2}$)')
        ax1.axis([0,12,10**13,3*10**15])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
        # plt.show()
        
        fig.savefig("../Acta Material Paper/Images/DislocationDensities/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def calc_discDensity(self):

        discDens = []
        for i,x in enumerate(self.X.tolist()[0]):

            Xfit = [self.Kscatt[j][i] for j in range(5) if self.KFWHMCorr[j][i] > 0.0]
            XfitCol = np.asarray(Xfit)[:,np.newaxis]
            Yfit = [self.dcosFWHMCorr[j][i] for j in range(5) if self.KFWHMCorr[j][i] > 0.0]

            if Yfit:
                slope, _, _, _ = np.linalg.lstsq(XfitCol, Yfit)
                line = slope*np.asarray(Xfit)
                discDenstemp = 14.4*(slope/(2.4768*10**-10))**2
                discDens.append([x, self.Y.tolist()[0][i], discDenstemp[0]])

            else:
                print "No data points to fit at x = ", x

        return discDens
                        
    def calc_K_ASTM(self):

        return 0.03162*(self.load/(self.B*math.sqrt(self.W)))*((2+(self.length/self.W))/(1-(self.length/self.W))**(3/2))* \
               (0.866+4.64*(self.length/self.W)-13.32*(self.length/self.W)**2+14.72*(self.length/self.W)**3-5.6*(self.length/self.W)**4) #MPa sqrt(m)
        
    def calc_K_StressFit(self):  

        linescan = [[self.X.tolist()[0][i], np.asarray(self.planeStrainSigmaYY)[0][i], np.asarray(self.planeStrainSigmaXX)[0][i]] \
                        for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01 and self.X.tolist()[0][i] > 0.1]

        X = [1/np.sqrt(i[0]) for i in linescan if i[0] > 0.1]
        
        errorPlotX = []
        errorPlotY = []
        
        def func(x, m):
            return m*x
        
        for ci in [5]:
            if self.length == 6.6+7.251 or self.length ==3.405+7.251 or self.length == 7.03+7.251:
                crit = 0.1
                critlow = 0.0
            else:
                if self.length == 4.3+7.251:
                    print "Enters this if"
                    crit = 0.1
                    critlow = 0.00
                else:
                    crit = ci * 1.0/100.0
                    critlow = 0.0
            testcond = [math.fabs((i[1]-i[2])/i[1]) for i in linescan]

            Xfit = [1/np.sqrt(i[0]) for jx,i in enumerate(linescan) if testcond[jx] < crit and testcond[jx] > critlow]
            Yfit = [i[1] for jy,i in enumerate(linescan) if testcond[jy] < crit and testcond[jy] > critlow]
           
            if Yfit:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xfit,Yfit)
                # popt, pcov = scipy.optimize.curve_fit(func,Xfit,Yfit)
                # perr = np.sqrt(np.diag(pcov))
                
                line = slope*np.asarray(Xfit) + intercept
                KStressFit = slope*np.sqrt(2*np.pi/1000)
                KStressFitError = std_err*np.sqrt(2*np.pi/1000)
                # slope = popt[0]
                # std_error = perr[0]
                # intercept = 0
                line = slope*np.asarray(Xfit) + intercept
                KStressFit = slope*np.sqrt(2*np.pi/1000)
                KStressFitError = std_err*np.sqrt(2*np.pi/1000)
                
                errorPlotX.append(crit)
                errorPlotY.append(std_err)
            else:
                print "No data points satisfy StressXX == StressYY to within ", crit*100, "% difference"
                KStressFit = 0
                KStressFitError = 0
                line = 0*np.asarray(Xfit) + 0
        
        return KStressFit, KStressFitError, Xfit, line

    def calc_PlasticZoneSize_Analytical(self):
        return (1/(3*np.pi))*1000*(self.KASTM/self.YieldStrength)**2  #mm

    def PlotStressVsSqrtR(self):
        def tick_function(x):
            V = (1/x**2)
            return ["%.2f" % z if (z<0.1) else "%.1f" % z if (z<1) else "%.0f" % z for z in V]

        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        
        errors = self.planeStraindSigmaYY
        color = colors[0]
        marker = markers[0]

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_sYYvsSqrtr"
        
        linescan = [[self.X.tolist()[0][i], np.asarray(self.planeStrainSigmaYY)[0][i], np.asarray(self.planeStraindSigmaYY)[0][i], \
                        np.asarray(self.planeStrainSigmaXX)[0][i], np.asarray(self.planeStraindSigmaXX)[0][i]] \
                            for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [1/np.sqrt(i[0]) for i in linescan if i[0] > 0.0]
        Y = [i[1] for i in linescan if i[0] > 0.0]
        StressXX_Y = [i[3] for i in linescan if i[0] > 0.0]
        StressXX_Yerr = [i[4] for i in linescan if i[0] > 0.0]
        Yerr = [i[2] for i in linescan if i[0] > 0.0]

###############################################################
## Plot Data
# Primary Axis
#        ax1.scatter(X,Y,color=color,marker=marker)
        ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='None', marker=marker, label = "StressYY")
        #ax1.errorbar(X, StressXX_Y, yerr = StressXX_Yerr, color=colors[1], marker=markers[1], label = "StressXX")
        ax1.legend(loc='upper right')
        ax1.set_xlabel(r'$\mathrm{r}^{-1/2}$ (mm $^{-1/2}$)')
        ax1.set_ylabel(r'Crack-opening stress, $\sigma_{YY}$ (MPa)')
        ax1.axis([0,6,-200,1200])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label, ax2.xaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels()):
            item.set_fontsize(16)
        
#Secondary Axis
        new_tick_locations = np.array([1/np.sqrt(2), 1/np.sqrt(1), 1/np.sqrt(0.5), 1/np.sqrt(0.2), 1/np.sqrt(0.1), 1/np.sqrt(0.05)])

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(r'$\mathrm{r}$ (mm)')
###############################################################
##Plot Fit Line
        ax1.plot(self.KStressFitLineX,self.KStressFitLineY, color="Red")
        
###############################################################
## Plot Plastic Zone Boundary (Analytical)

        ax1.axvline(x=1/np.sqrt(self.PZsize_Analytical), color=colors[0], linestyle='dashed')
        ax1.text(1/np.sqrt(self.PZsize_Analytical) + 0.15,800, r'$\mathrm{r}_p$',fontsize=18) 
        # if(self.load == 1700):
            # plt.show()

        fig.savefig("../Acta Material Paper/Images/Stresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotAllStrains(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_Strain"

        for e in [self.epsXX, self.epsYY, self.epsXY]:
            if np.array_equal(e,self.epsXX):
                errors = self.depsXX
                label = r'$\epsilon_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.epsYY):
                    errors = self.depsYY
                    label = r'$\epsilon_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                    
                else:
                    if np.array_equal(e,self.epsXY):
                        errors = self.depsXY
                        label = r'$\epsilon_{XY}$'
                        color = colors[2]
                        marker = markers[2]                    

    #####
    ## Plot strain
    #####   
 
            linescan = [[self.X.tolist()[0][i], e.tolist()[0][i], errors.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]

            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]
            Yerr = [i[2] for i in linescan]

#            ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
            ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
            ax1.legend(loc='upper right')
            ax1.set_title("Strain")    
            ax1.set_xlabel('Distance from crack tip (mm)')
            ax1.set_ylabel(r'Strain / $10^-6$')
            ax1.axis([0,12,-500,3000])
            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                         ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
 
        # plt.show()
        fig.savefig("../Acta Material Paper/Images/Strains/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotAllFWHMsVsX(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_FWHM"

        for ri, e in enumerate(self.reflections):
            
            data = self.KFWHMCorr[ri]
            # print data
            label = e
            color = colors[ri]
            marker = markers[ri]

    #####
    ## Plot strain
    #####   
 
            linescan = [[self.X.tolist()[0][i], data[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]

            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]
            # Yerr = [i[2] for i in linescan]

            ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
            # ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
            ax1.legend(loc='upper right')
            ax1.set_title("FWHM")    
            ax1.set_xlabel('Distance from crack tip (mm)')
            ax1.set_ylabel(r'FWHM (nm$^-1$)')
            ax1.axis([0,12,0.0,0.02])
            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                         ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
 
        # plt.show()
        # fig.savefig("../Acta Material Paper/Images/FWHMs/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotAllStresses(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_Stress"

        for e in [self.planeStrainSigmaXX, self.planeStrainSigmaYY, self.planeStrainSigmaXY, self.planeStrainSigmaZZ]:
            if np.array_equal(e,self.planeStrainSigmaXX):
                label = r'$\sigma_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.planeStrainSigmaYY):
                    label = r'$\sigma_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                    
                else:
                    if np.array_equal(e,self.planeStrainSigmaXY):
                        label = r'$\sigma_{XY}$'
                        color = colors[2]
                        marker = markers[2]                    

                    else:
                        if np.array_equal(e,self.planeStrainSigmaZZ):
                            label = r'$\sigma_{ZZ}$'
                            color = colors[3]
                            marker = markers[3]                           
                        
    #####
    ## Plot strain
    #####   
 
            linescan = [[self.X.tolist()[0][i], e.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]

            ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
            ax1.legend(loc='upper right')
            ax1.set_title("Stress")    
            ax1.set_xlabel('Distance from crack tip (mm)')
            ax1.set_ylabel(r'Stress (MPa)')
            ax1.axis([0,12,-500,1200])
            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                         ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
                
        # plt.show()
        fig.savefig("../Acta Material Paper/Images/Stresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotAllStrainsContour(self):

        for e in [self.epsXX, self.epsYY, self.epsXY]:
            if np.array_equal(e,self.epsXX):

                label = r'$\epsilon_{XX}$'
                color = colors[0]
                marker = markers[0]
                
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_StrainXXContour"
                
                #####
                ## Plot strain XX
                #####        

                linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.epsXX.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0])]
                X = np.asarray([i[0] for i in linescan])
                Y = np.asarray([i[1] for i in linescan])
                Z = np.asarray([i[2] for i in linescan])

                xi = np.linspace(X.min(), X.max(), 100)
                yi = np.linspace(Y.min(), Y.max(), 100)
                zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
            
                sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = 3000, vmin = -500)

                plt.axes().set_aspect(aspect="auto")
                
                ax1.set_title("Strain XX")    
                ax1.set_xlabel('Distance from crack tip X (mm)')
                ax1.set_ylabel('Distance from crack tip Y (mm)')

                plt.colorbar(sc)
        
                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_fontsize(18)
                    
#                plt.show()
                fig.savefig("../Acta Material Paper/Images/Strains/"+fileLabel+".pdf")
                plt.close(fig)
                        
            else:
                if np.array_equal(e,self.epsYY):

                    label = r'$\epsilon_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                    
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_StrainYYContour"
                    
                    #####
                    ## Plot strain XX
                    #####        
                    
                    linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.epsYY.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0])]
                    X = np.asarray([i[0] for i in linescan])
                    Y = np.asarray([i[1] for i in linescan])
                    Z = np.asarray([i[2] for i in linescan])

                    xi = np.linspace(X.min(), X.max(), 100)
                    yi = np.linspace(Y.min(), Y.max(), 100)
                    zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
                
                    sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = 3000, vmin = -500)

                    plt.axes().set_aspect(aspect="auto")
                    
                    ax1.set_title("Strain YY")    
                    ax1.set_xlabel('Distance from crack tip X (mm)')
                    ax1.set_ylabel('Distance from crack tip Y (mm)')

                    plt.colorbar(sc)

                    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                 ax1.get_xticklabels() + ax1.get_yticklabels()):
                        item.set_fontsize(18)
                        
#                    plt.show()
                    fig.savefig("../Acta Material Paper/Images/Strains/"+fileLabel+".pdf")
                    plt.close(fig)
             
                else:
                    if np.array_equal(e,self.epsXY):
                        label = r'$\epsilon_{XY}$'
                        color = colors[2]
                        marker = markers[2]
                        
                        fig = plt.figure()
                        ax1 = fig.add_subplot(111)

                        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_StrainXYContour"
                        
                        #####
                        ## Plot strain XX
                        #####        
                        
                        linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.epsXY.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0])]
                        X = np.asarray([i[0] for i in linescan])
                        Y = np.asarray([i[1] for i in linescan])
                        Z = np.asarray([i[2] for i in linescan])

                        xi = np.linspace(X.min(), X.max(), 100)
                        yi = np.linspace(Y.min(), Y.max(), 100)
                        zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
                    
                        sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = 3000, vmin = -500)

                        plt.axes().set_aspect(aspect="auto")
                        
                        ax1.set_title("Strain XY")    
                        ax1.set_xlabel('Distance from crack tip X (mm)')
                        ax1.set_ylabel('Distance from crack tip Y (mm)')

                        plt.colorbar(sc)

                        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                     ax1.get_xticklabels() + ax1.get_yticklabels()):
                            item.set_fontsize(18)
                            
#                        plt.show()
                        fig.savefig("../Acta Material Paper/Images/Strains/"+fileLabel+".pdf",bbox_inches='tight')
                        plt.close(fig)
    
    def PlotTheta(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_Theta"

        errors = self.dtheta
        label = r'$\theta$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot triaxiality
    #####        
        linescan = [[self.X.tolist()[0][i], self.theta.tolist()[0][i], errors.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1]*180.0/math.pi for i in linescan]
        Yerr = [i[2]*180.0/math.pi for i in linescan]
        
        # ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
        ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
        ax1.set_title("Principal Axis Orientation")    
        ax1.set_xlabel('Distance from crack tip (mm)')
        ax1.set_ylabel(r'$\theta$')
        ax1.axis([0,12,0,180])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
                
    #        plt.show()
        fig.savefig("../Acta Material Paper/Images/Theta/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
    
    def PlotAllPrincipalStrains(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStrain"

        for e in [self.PrincipalStrainXX, self.PrincipalStrainYY]:
            if np.array_equal(e,self.PrincipalStrainXX):
                errors = self.dPrincipalStrainXX
                label = r'$\epsilon\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.PrincipalStrainYY):
                    errors = self.dPrincipalStrainYY
                    label = r'$\epsilon\prime_{YY}$'
                    color = colors[1]
                    marker = markers[1]

    #####
    ## Plot strain
    #####   
 
            linescan = [[self.X.tolist()[0][i], e[i], errors.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]
            Yerr = [i[2] for i in linescan]
            
            # ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
            ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
            ax1.legend(loc='upper right')
            ax1.set_title("Principal Strain")    
            ax1.set_xlabel('Distance from crack tip (mm)')
            ax1.set_ylabel(r'Principal Strain / $10^-6$')
            ax1.axis([0,12,0,2500])
            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                         ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
                
    #        plt.show()
        fig.savefig("../Acta Material Paper/Images/PrincipalStrains/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
     
    def PlotAllPrincipalStrainsContour(self):

        for e in [self.PrincipalStrainXX, self.PrincipalStrainYY]:
            if np.array_equal(e,self.PrincipalStrainXX):
                    
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                label = r'$\epsilon\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
                
                fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStrainXXContour"
                
                #####
                ## Plot strain XX
                #####        
                
                linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.PrincipalStrainXX[i]] for i,j in enumerate(self.Y.tolist()[0])]
                X = np.asarray([i[0] for i in linescan])
                Y = np.asarray([i[1] for i in linescan])
                Z = np.asarray([i[2] for i in linescan])

                xi = np.linspace(X.min(), X.max(), 100)
                yi = np.linspace(Y.min(), Y.max(), 100)
                zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
            
                sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = 3000, vmin = -500)

                plt.axes().set_aspect(aspect="auto")
                
                ax1.legend(loc='upper right')
                ax1.set_title("Principal Strains XX")    
                ax1.set_xlabel('Distance from crack tip X (mm)')
                ax1.set_ylabel('Distance from crack tip Y (mm)')

                plt.colorbar(sc)
        
                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_fontsize(18)
                    
#                plt.show()
                fig.savefig("../Acta Material Paper/Images/PrincipalStrains/"+fileLabel+".pdf",bbox_inches='tight')
                plt.close(fig)
                    
            else:
                if np.array_equal(e,self.PrincipalStrainYY):
                    
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)

                    label = r'$\epsilon\prime_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                    
                    fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStrainYYContour"
                    
                    #####
                    ## Plot strain YY
                    #####        
                    
                    linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.PrincipalStrainYY[i]] for i,j in enumerate(self.Y.tolist()[0])]
                    X = np.asarray([i[0] for i in linescan])
                    Y = np.asarray([i[1] for i in linescan])
                    Z = np.asarray([i[2] for i in linescan])

                    xi = np.linspace(X.min(), X.max(), 100)
                    yi = np.linspace(Y.min(), Y.max(), 100)
                    zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
                
                    sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = 3000, vmin = -500)

                    plt.axes().set_aspect(aspect="auto")
                    
                    ax1.legend(loc='upper right')
                    ax1.set_title("Principal Strain YY")    
                    ax1.set_xlabel('Distance from crack tip X (mm)')
                    ax1.set_ylabel('Distance from crack tip Y (mm)')

                    plt.colorbar(sc)
            
                    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                 ax1.get_xticklabels() + ax1.get_yticklabels()):
                        item.set_fontsize(18)
                        
#                    plt.show()
                    fig.savefig("../Acta Material Paper/Images/PrincipalStrains/"+fileLabel+".pdf",bbox_inches='tight')
                    plt.close(fig)
                
    def PlotAllPrincipalStresses(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStress"

        for e in [self.PrincipalStressXX, self.PrincipalStressYY, self.PrincipalStressZZ]:
            if np.array_equal(e,self.PrincipalStressXX):
                errors = self.dPrincipalStressXX
                label = r'$\sigma\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.PrincipalStressYY):
                    errors = self.dPrincipalStressYY
                    label = r'$\sigma\prime_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                else:
                    if np.array_equal(e,self.PrincipalStressZZ):
                        errors = self.dPrincipalStressZZ
                        label = r'$\sigma\prime_{ZZ}$'
                        color = colors[2]
                        marker = markers[2]
    #####
    ## Plot strain
    #####        
            linescan = [[self.X.tolist()[0][i], e[i], errors.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]
            Yerr = [i[2] for i in linescan]

            # ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
            ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
            ax1.legend(loc='upper right')
            ax1.set_title("Principal Stress")    
            ax1.set_xlabel('Distance from crack tip (mm)')
            ax1.set_ylabel('Principal Stress (MPa)')
            ax1.axis([0,12,-200,1200])
            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                         ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
                
    #        plt.show()
        fig.savefig("../Acta Material Paper/Images/PrincipalStresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
     
    def PlotAllPrincipalStressesContour(self):
    
        for e in [self.PrincipalStressXX, self.PrincipalStressYY, self.PrincipalStressZZ]:
            if np.array_equal(e,self.PrincipalStressXX):
                
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                label = r'$\sigma\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
                
                fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStressXXContour"
                
                #####
                ## Plot stress XX
                #####        
                
                linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.PrincipalStressXX[i]] for i,j in enumerate(self.Y.tolist()[0])]
                X = np.asarray([i[0] for i in linescan])
                Y = np.asarray([i[1] for i in linescan])
                Z = np.asarray([i[2] for i in linescan])

                xi = np.linspace(X.min(), X.max(), 100)
                yi = np.linspace(Y.min(), Y.max(), 100)
                zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
            
                sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = -200, vmin = 1200)

                plt.axes().set_aspect(aspect="auto")
                
                ax1.legend(loc='upper right')
                ax1.set_title("Principal Stress XX")    
                ax1.set_xlabel('Distance from crack tip X (mm)')
                ax1.set_ylabel('Distance from crack tip Y (mm)')

                plt.colorbar(sc)
        
                for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                    item.set_fontsize(18)
                    
                # plt.show()
                fig.savefig("../Acta Material Paper/Images/PrincipalStresses/"+fileLabel+".pdf",bbox_inches='tight')
                plt.close(fig)
                
            else:
                if np.array_equal(e,self.PrincipalStressYY):

                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    
                    label = r'$\sigma\prime_{YY}$'
                    color = colors[0]
                    marker = markers[0]
                    
                    fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStressYYContour"
                    
                    #####
                    ## Plot stress XX
                    #####        
                    
                    linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.PrincipalStressYY[i]] for i,j in enumerate(self.Y.tolist()[0])]
                    X = np.asarray([i[0] for i in linescan])
                    Y = np.asarray([i[1] for i in linescan])
                    Z = np.asarray([i[2] for i in linescan])

                    xi = np.linspace(X.min(), X.max(), 100)
                    yi = np.linspace(Y.min(), Y.max(), 100)
                    zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
                
                    sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = -200, vmin = 1200)

                    plt.axes().set_aspect(aspect="auto")
                
                    ax1.legend(loc='upper right')
                    ax1.set_title("Principal Stress YY")    
                    ax1.set_xlabel('Distance from crack tip X (mm)')
                    ax1.set_ylabel('Distance from crack tip Y (mm)')

                    plt.colorbar(sc)
        
                    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                 ax1.get_xticklabels() + ax1.get_yticklabels()):
                        item.set_fontsize(18)
                        
                    # plt.show()
                    fig.savefig("../Acta Material Paper/Images/PrincipalStresses/"+fileLabel+".pdf",bbox_inches='tight')
                    plt.close(fig)
                else:
                    if np.array_equal(e,self.PrincipalStressZZ):

                        fig = plt.figure()
                        ax1 = fig.add_subplot(111)

                        label = r'$\sigma\prime_{ZZ}$'
                        color = colors[0]
                        marker = markers[0]
                        
                        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStressZZContour"
                        
                        #####
                        ## Plot stress XX
                        #####        
                        
                        linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.PrincipalStressZZ[i]] for i,j in enumerate(self.Y.tolist()[0])]
                        X = np.asarray([i[0] for i in linescan])
                        Y = np.asarray([i[1] for i in linescan])
                        Z = np.asarray([i[2] for i in linescan])

                        xi = np.linspace(X.min(), X.max(), 100)
                        yi = np.linspace(Y.min(), Y.max(), 100)
                        zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
                    
                        sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = -200, vmin = 1200)

                        plt.axes().set_aspect(aspect="auto")
                
                        ax1.legend(loc='upper right')
                        ax1.set_title("Principal Stress ZZ")    
                        ax1.set_xlabel('Distance from crack tip X (mm)')
                        ax1.set_ylabel('Distance from crack tip Y (mm)')

                        plt.colorbar(sc)
        
                        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                                     ax1.get_xticklabels() + ax1.get_yticklabels()):
                            item.set_fontsize(18)
                            
                        # plt.show()
                        fig.savefig("../Acta Material Paper/Images/PrincipalStresses/"+fileLabel+".pdf",bbox_inches='tight')
                        plt.close(fig)
    
    def PlotvonMisesStress(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_vonMisesStress"

        error = self.dvonMisesStress
        label = r'$\sigma\prime_{VM}$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot vonMisesStress
    #####        
        linescan = [[self.X.tolist()[0][i], self.vonMisesStress[i], error[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]
        Yerr = [i[2] for i in linescan]

        # ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
        ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)       
        ax1.set_title("von Mises Stress")    
        ax1.set_xlabel('Distance from crack tip (mm)')
        ax1.set_ylabel('von Mises Stress (MPa)')
        ax1.axis([0,12,0,400])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
                
    #        plt.show()
        fig.savefig("../Acta Material Paper/Images/vonMisesStresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
  
    def PlotvonMisesStressContour(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_vonMisesStressContour"

        label = r'$\sigma\prime_{VM}$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot vonMisesStress
    #####        
        linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.vonMisesStress[i]] for i,j in enumerate(self.Y.tolist()[0])]
        X = np.asarray([i[0] for i in linescan])
        Y = np.asarray([i[1] for i in linescan])
        Z = np.asarray([i[2] for i in linescan])

        xi = np.linspace(X.min(), X.max(), 100)
        yi = np.linspace(Y.min(), Y.max(), 100)
        zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
        
        sc = ax1.imshow(zi, extent = [X.min(), 4, Y.min(), Y.max()], vmax = 400, vmin = 0)

        plt.axes().set_aspect(aspect="auto")
                
        ax1.set_title("von Mises Stress")    
        ax1.set_xlabel('Distance from crack tip X (mm)')
        ax1.set_ylabel('Distance from crack tip Y (mm)')

        plt.colorbar(sc)
        
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
                
        # plt.show()
        fig.savefig("../Acta Material Paper/Images/vonMisesStresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
     
    def PlotHydrostaticStress(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_HydrostaticStress"

        errors = self.dHydrostaticStress
        label = r'$\sigma_H$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot hydrostatic stress
    #####
    
        linescan = [[self.X.tolist()[0][i], self.HydrostaticStress[i], self.dHydrostaticStress.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]
        Yerr = [i[2] for i in linescan]

        # ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
        ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
        ax1.set_title("Hydrostatic Stress")    
        ax1.set_xlabel('Distance from crack tip (mm)')
        ax1.set_ylabel(r'Hydrostatic Stress $\sigma_H$')
        ax1.axis([0,1,0,1000])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
                
    #        plt.show()
        fig.savefig("../Acta Material Paper/Images/Stresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
    
    def PlotTriaxiality(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_Triaxiality"

        errors = self.deta
        label = r'$\eta$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot triaxiality
    #####        
        linescan = [[self.X.tolist()[0][i], self.eta[i], errors[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]
        Yerr = [i[2] for i in linescan]

        # ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
        ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)
        ax1.set_title("Stress Triaxiality")    
        ax1.set_xlabel('Distance from crack tip (mm)')
        ax1.set_ylabel(r'Triaxiality $\eta$')
        ax1.axis([0,12,0,4])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
                
    #        plt.show()
        fig.savefig("../Acta Material Paper/Images/StressTriaxiality/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotTriaxialityContour(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_TriaxialityContour"

        label = r'$\eta$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot triaxiality
    #####        

        linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.eta[i]] for i,j in enumerate(self.Y.tolist()[0])]
        X = np.asarray([i[0] for i in linescan])
        Y = np.asarray([i[1] for i in linescan])
        Z = np.asarray([i[2] for i in linescan])
        
        xi = np.linspace(X.min(), X.max(), 100)
        yi = np.linspace(Y.min(), Y.max(), 100)
        zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
        
        sc = ax1.imshow(zi, extent = [X.min(), X.max(), Y.min(), Y.max()], vmax = 4, vmin = 0)

        plt.axes().set_aspect(aspect="auto")
        
        ax1.set_title("Stress Triaxiality")    
        ax1.set_xlabel('Distance from crack tip X (mm)')
        ax1.set_ylabel('Distance from crack tip Y (mm)')

        plt.colorbar(sc)
        
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
        
        fig.savefig("../Acta Material Paper/Images/StressTriaxiality/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
    
    def PlotDislocationDensity(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"DislocationDensity"

        label = r'$\eta$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot triaxiality
    #####        
        linescan = [[self.discX.tolist()[0][i], self.discDens.tolist()[0][i]] for i,j in enumerate(c.discDens.tolist()[0]) if j > 0.0]
        X = [i[0] for i in linescan]
        Y = [i[1]/(10**14) for i in linescan]

        ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
        ax1.set_title(r'Dislocation Density')    
        ax1.set_xlabel('Distance from crack tip (mm)')
        ax1.set_ylabel(r'Dislocation Density $\rho$ (m $^{-2}$ / 10$^{14}$)')
        ax1.axis([0,12,0,30])
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
                
        # plt.show()
        fig.savefig("../Acta Material Paper/Images/DislocationDensities/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotDislocationDensityContour(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_DislocationDensityContour"

        label = r'$\eta$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot triaxiality
    #####        

        linescan = [[self.X.tolist()[0][i], self.Y.tolist()[0][i], self.eta[i]] for i,j in enumerate(self.Y.tolist()[0])]
        X = np.asarray([i[0] for i in linescan])
        Y = np.asarray([i[1] for i in linescan])
        Z = np.asarray([i[2] for i in linescan])
        
        xi = np.linspace(X.min(), X.max(), 100)
        yi = np.linspace(Y.min(), Y.max(), 100)
        zi = plt.mlab.griddata(X, Y, Z, xi, yi, interp='linear')
        
        sc = ax1.imshow(zi, extent = [X.min(), X.max(), Y.min(), Y.max()], vmax = 4, vmin = 0)

        plt.axes().set_aspect(aspect="auto")
        
        ax1.set_title("Dislocation Density Contour")    
        ax1.set_xlabel('Distance from crack tip X (mm)')
        ax1.set_ylabel('Distance from crack tip Y (mm)')

        plt.colorbar(sc)
        
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(18)
        
        fig.savefig("../Acta Material Paper/Images/StressTriaxiality/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)
#########################################################
################ End of Crack Class #####################
#########################################################

## Some functions for plotting comparisons between Air and Hydrogen Cracks
      
def PlotStrainYYComparison(aircrack, Hcrack):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    fileLabel = "Comparison_" + aircrack.env + "_" + str(aircrack.load) + "_a" + str(round(aircrack.length,2)) + "_K" + str(round(aircrack.KASTM,2))+ \
                    + Hcrack.env + "_" + str(Hcrack.load) + "_a" + str(round(Hcrack.length,2)) + "_K" + str(round(Hcrack.KASTM,2))+"_StrainYY"

    for i, c in enumerate([aircrack, Hcrack])
        
        e = c.epsYY
        errors = c.depsYY
        label = c.env
        color = colors[i]
        marker = markers[i]

        linescan = [[c.X.tolist()[0][i], e.tolist()[0][i], errors.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]

        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]
        Yerr = [i[2] for i in linescan]

        ax1.errorbar(X,Y,yerr=Yerr, color=color, linestyle='-', marker=marker, label=label)

    ax1.legend(loc='upper right')
    ax1.set_title("Strain")    
    ax1.set_xlabel('Distance from crack tip (mm)')
    ax1.set_ylabel(r'Strain / $10^-6$')
    ax1.axis([0,12,-500,3000])
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    # plt.show()
    fig.savefig("../Acta Material Paper/Images/Strains/"+fileLabel+".pdf",bbox_inches='tight')
    plt.close(fig)

def PlotDislocationDensityComparison(aircrack, Hcrack):    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    fileLabel = "Comparison_" + aircrack.env + "_" + str(aircrack.load) + "_a" + str(round(aircrack.length,2)) + "_K" + str(round(aircrack.KASTM,2))+ \
                    + Hcrack.env + "_" + str(Hcrack.load) + "_a" + str(round(Hcrack.length,2)) + "_K" + str(round(Hcrack.KASTM,2))+"_DislocationDensity"

    for i, c in enumerate([aircrack, Hcrack])
        
        label = c.env
        color = colors[i]
        marker = markers[i]

        linescan = [[j[0], j[2]] for j in c.discDens if j[1] > -0.01 and j[1] < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]
        
        ax1.scatter(X, Y)

    ax1.set_title("Dislocation Density")    
    ax1.set_xlabel('Distance from crack tip (mm)')
    ax1.set_ylabel('Dislocation Density (mm$^{-2}$)')
    ax1.axis([0,12,10**13,3*10**15])
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    # plt.show()
    fig.savefig("../Acta Material Paper/Images/DislocationDensities/"+fileLabel+".pdf",bbox_inches='tight')
    plt.close(fig)
    
def PlotKvsa(cracklist):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    fileLabel = "KappVsa"

    aircracks = [c for c in cracklist if c.env == "Air"]
    H2cracks = [c for c in cracklist if c.env == "Hydrogen"]
    

## Plot KASTM function
    def KASTM(a,B,W,P):
        return 0.03162*(P/(B*math.sqrt(W)))*((2+(a/W))/(1-(a/W))**(3/2))*(0.866+4.64*(a/W)-13.32*(a/W)**2+14.72*(a/W)**3-5.6*(a/W)**4)

    X = range(0,30)
    Y = [KASTM(xi,cracklist[0].B, cracklist[0].W, 1700) for xi in X]
    
    color = colors[2]
    marker = 'None'
    label = 'KASTM'
    ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)

## Plot Hydrogen Data
    X = []
    Y = []
    Yerr = []
    for hc in H2cracks:
        X.append(hc.length)
        Y.append(hc.KStressFit)
        Yerr.append(hc.KStressFitError)

    label = "Hydrogen"
    color = colors[0]
    marker = markers[0]
    ax1.errorbar(X,Y,Yerr,color=color,linestyle='None',marker=marker,label=label)

## Plot Air Data
    X = []
    Y = []
    Yerr = []
    for ac in aircracks:
        X.append(ac.length)
        Y.append(ac.KStressFit)
        Yerr.append(ac.KStressFitError)

    label = "Air"
    color = colors[1]
    marker = markers[1]
    ax1.errorbar(X,Y,Yerr,color=color,linestyle='None',marker=marker,label=label)
    
    ax1.legend(loc='lower right')
    ax1.set_title(r'K vs Crack Length')    
    ax1.set_xlabel('Crack Length (mm)')
    ax1.set_ylabel(r'K$_{app}$ (MPa m $^{1/2}$)')

    ax1.axis([5,20,0,60])
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)
            
    # plt.show()
    fig.savefig("../Acta Material Paper/Images/"+fileLabel+".pdf",bbox_inches='tight')
    plt.close(fig)
       
def PlotKvsKASTM(cracklist):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    fileLabel = "KeffVsKapp"

    aircracks = [c for c in cracklist if c.env == "Air"]
    H2cracks = [c for c in cracklist if c.env == "Hydrogen"]
    

## Plot KASTM function
    def KASTM(a,B,W,P):
        return 0.03162*(P/(B*math.sqrt(W)))*((2+(a/W))/(1-(a/W))**(3/2))*(0.866+4.64*(a/W)-13.32*(a/W)**2+14.72*(a/W)**3-5.6*(a/W)**4)

    X = range(0,30)
    Y = [KASTM(xi,cracklist[0].B, cracklist[0].W, 1700) for xi in X]
    X = Y
    color = colors[2]
    marker = 'None'
    label = 'KASTM'
    ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)

## Plot Hydrogen Data
    X = []
    Y = []
    Yerr = []
    for hc in H2cracks:
        X.append(KASTM(hc.length,hc.B,hc.W,hc.load))
        Y.append(hc.KStressFit)
        Yerr.append(hc.KStressFitError)

    label = "Hydrogen"
    color = colors[0]
    marker = markers[0]
    ax1.errorbar(X,Y,Yerr,color=color,linestyle='None',marker=marker,label=label)

## Plot Air Data
    X = []
    Y = []
    Yerr = []
    for ac in aircracks:
        X.append(KASTM(ac.length,ac.B,ac.W,ac.load))
        Y.append(ac.KStressFit)
        Yerr.append(ac.KStressFitError)

    label = "Air"
    color = colors[1]
    marker = markers[1]
    ax1.errorbar(X,Y,Yerr,color=color,linestyle='None',marker=marker,label=label)
    
    ax1.legend(loc='lower right')
    ax1.set_title(r'Keff vs Kapplied')    
    ax1.set_xlabel(r'K$_{applied}$ (MPa m $^{1/2}$)')
    ax1.set_ylabel(r'K$_{effective}$ (MPa m $^{1/2}$)')

    ax1.axis([0,40,0,60])
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)
            
    # plt.show()
    fig.savefig("../Acta Material Paper/Images/"+fileLabel+".pdf",bbox_inches='tight')
    plt.close(fig)    

## Function for writing all data to .xlsx spreadsheet
def writeDatatoXLSX(cracklist):

    xlslabels = ["Distance Ahead of Crack Tip (mm)", "StrainXX (10^6 m/m)", "StrainYY (10^6 m/m)", "StrainXY  (10^6 m/m)", "StressXX (MPa)", "StressYY (MPa)", "StressXY (MPa)", "StressZZ (MPa)",
                 "StrainXX'  (10^6 m/m)", "StrainYY'  (10^6 m/m)", "StressXX' (MPa)", "StressYY' (MPa)", "StressZZ' (MPa)", "von Mises Stress (MPa)", "Triaxality", "Hydostatic Stress (MPa)"]
    xlsdf = pd.DataFrame(columns=xlslabels)

# Create a pandas excel writer using xlsxWriter
    writer = pd.ExcelWriter('AllData.xlsx', engine='xlsxwriter')
    
    for c in cracklist:
    
        SheetLabel = c.env + "_P" + str(c.load) + "_a" + str(c.length) + "_k" + str(round(c.KASTM,1))
   
#       Turn each dataset into Pandas dataframe
        for qi,e in enumerate([c.epsXX, c.epsYY, c.epsXY, c.planeStrainSigmaXX, c.planeStrainSigmaYY, c.planeStrainSigmaXY, c.planeStrainSigmaZZ,
                  c.PrincipalStrainXX, c.PrincipalStrainYY, c.PrincipalStressXX, c.PrincipalStressYY, c.PrincipalStressZZ, 
                  c.vonMisesStress, c.eta, c.HydrostaticStress]):
                  
            if type(e) == list:
                linescan = [[c.X.tolist()[0][i], e[i]] for i,j in enumerate(c.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            else:
                if type(e) == np.ndarray and e.ndim == 1:
                    linescan = [[c.X.tolist()[0][i], e.tolist()[i]] for i,j in enumerate(c.Y.tolist()[0]) if j > -0.01 and j < 0.01]
                else:
                    linescan = [[c.X.tolist()[0][i], e.tolist()[0][i]] for i,j in enumerate(c.Y.tolist()[0]) if j > -0.01 and j < 0.01]

            X = np.asarray([i[0] for i in linescan])
            Y = np.asarray([i[1] for i in linescan])

            if qi==0:
                xlsdf[xlslabels[0]] = X
            xlsdf[xlslabels[qi+1]] = Y

        xlsdf.to_excel(writer, sheet_name = SheetLabel)
        
    writer.save()
        
###################################################################################################
################################## Main program ###################################################
###################################################################################################
       
#Create crack instances
B = 3.0
W = 26.67
crackList = []

######Hydrogen Cracks #####
crackList.append({'Condition':"Hydrogen", 'Load':1700,'crackPosX':3.085, 'crackPosY':0.09, 'dataFile':"H2_Crack0_Load1700_683_929_fit_5_22-Dec-2017.txt", 'discfile':"Crack1_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':3.275, 'crackPosY':0.075, 'dataFile':"H2_Crack1_Load850_1179_1425_fit_5_22-Dec-2017.txt", 'discfile':"Crack1_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700,'crackPosX':3.275, 'crackPosY':0.15, 'dataFile':"H2_Crack1_Load1700_931_1177_fit_5_22-Dec-2017.txt", 'discfile':"Crack1_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':3.405, 'crackPosY':-0.015, 'dataFile':"H2_Crack2_Load850_1675_1921_fit_5_22-Dec-2017.txt", 'discfile':"Crack2_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':3.405, 'crackPosY':-0.080, 'dataFile':"H2_Crack2_Load1700_1427_1673_fit_5_22-Dec-2017.txt", 'discfile':"Crack2_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':3.80, 'crackPosY':0.015, 'dataFile':"H2_Crack3_Load850_2171_2417_fit_5_22-Dec-2017.txt", 'discfile':"Crack3_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':3.80, 'crackPosY':-0.055, 'dataFile':"H2_Crack3_Load1700_1923_2169_fit_5_22-Dec-2017.txt", 'discfile':"Crack3_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':4.25, 'crackPosY':0.025, 'dataFile':"H2_Crack4_Load850_2667_2913_fit_5_22-Dec-2017.txt", 'discfile':"Crack4_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':4.25, 'crackPosY':-0.04, 'dataFile':"H2_Crack4_Load1700_2419_2665_fit_5_22-Dec-2017.txt", 'discfile':"Crack4_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':5.3, 'crackPosY':0.05, 'dataFile':"H2_Crack5_Load850_3163_3409_fit_5_22-Dec-2017.txt", 'discfile':"Crack5_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':5.3, 'crackPosY':-0.039, 'dataFile':"H2_Crack5_Load1700_2915_3161_fit_5_22-Dec-2017.txt", 'discfile':"Crack5_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':6.185, 'crackPosY':0.035, 'dataFile':"H2_Crack6_Load850_3660_3906_fit_5_22-Dec-2017.txt", 'discfile':"Crack6_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':6.185, 'crackPosY':-0.042, 'dataFile':"H2_Crack6_Load1700_3412_3658_fit_5_22-Dec-2017.txt", 'discfile':"Crack6_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':6.60, 'crackPosY':0.045, 'dataFile':"H2_Crack7_Load850_4156_4402_fit_5_20-Dec-2017.txt", 'discfile':"Crack7_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':6.60, 'crackPosY':0.048, 'dataFile':"H2_Crack7_Load1700_3908_4154_fit_5_05-Feb-2018.txt", 'discfile':"Crack7_Load1700.csv"}) 
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':7.03, 'crackPosY':0.05, 'dataFile':"H2_Crack8_Load850_4652_4898_fit_5_21-Feb-2018.txt", 'discfile':"Crack8_Load850.csv"})
crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':7.03, 'crackPosY':-0.055, 'dataFile':"H2_Crack8_Load1700_4404_4650_fit_5_21-Feb-2018.txt", 'discfile':"Crack8_Load1700.csv"})

#####Air Cracks #####
# crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':2.70, 'crackPosY':-0.015, 'dataFile':"Air_Crack0_Load850_5154_5400_fit_5_27-Dec-2017.txt", 'discfile':"Air_Crack1_Load850.csv"})
crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':2.70, 'crackPosY':-0.284995, 'dataFile':"Air_Crack0_Load1700_5402_5648_fit_5_27-Dec-2017.txt", 'discfile':"Air_Crack1_Load850.csv"})
# crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':4.3, 'crackPosY':0.1, 'dataFile':"Air_Crack1_Load850_5898_6144_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack1_Load850.csv"})
crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':4.3, 'crackPosY':0.04, 'dataFile':"Air_Crack1_Load1700_5650_5896_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack1_Load1700.csv"})
# crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':5.12, 'crackPosY':0.135, 'dataFile':"Air_Crack2_Load850_6394_6640_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack2_Load850.csv"})
crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':5.12, 'crackPosY':0.1, 'dataFile':"Air_Crack2_Load1700_6146_6392_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack2_Load1700.csv"})
# crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':6.36, 'crackPosY':0.15, 'dataFile':"Air_Crack3_Load850_6890_7136_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack3_Load850.csv"})
crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':6.36, 'crackPosY':0.08, 'dataFile':"Air_Crack3_Load1700_6642_6888_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack3_Load1700.csv"})

crack = []
for i,crackDict in enumerate(crackList):
    if(crackDict["Condition"]=="Hydrogen"):
        d0 = 2.0136519
    else:
        d0 = 1.996372929
    crack.append(Crack(crackDict["Condition"],crackDict["Load"],B,W,d0,7.251+crackDict["crackPosX"],crackDict["crackPosX"],crackDict["crackPosY"],crackDict["dataFile"],crackDict["discfile"]))
    
#####################################################################
#Do some plotting
#####################################################################

# PmaxCracks = [c for c in crack if c.load == 1700]
# writeDatatoXLSX(crack)    
for i,c in enumerate(crack):
    c.plot_discDensity()
    # c.PlotTheta()
    # print c.env, c.load, c.length, c.KASTM
    # c.PlotStressVsSqrtR()  
    # c.PlotAllStrains()
    # c.PlotAllPrincipalStrains()
    # c.PlotTheta()
    # c.PlotAllPrincipalStresses()
    # c.PlotvonMisesStress()
    # c.PlotTriaxiality()
    # c.PlotTriaxialityContour()
    # c.PlotvonMisesStressContour()
    # c.PlotAllStrainsContour()
    # c.PlotHydrostaticStress()
    # c.PlotAllStresses()
# PlotKvsa(PmaxCracks)
# PlotKvsKASTM(PmaxCracks)
PlotStrainYYComparison()