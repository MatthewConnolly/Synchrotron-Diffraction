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
    # datadir = "/home/matt/Q/Experiments/1-ID Summer 2017/Data/1-ID Summer 2017 Tomo Files/connolly_jun17/matlab/Analysis/"
    # dislocationdir = "/home/matt/Q/Experiments/1-ID Summer 2017/Acta Material Paper/Images/DislocationData/"

    datadir = "Example Data/"
    dislocationdir = datadir
    
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
        self.X, self.Y, self.epsXX, self.depsXX, self.epsYY, self.depsYY, self.epsXY, self.depsXY = self.set_Strain()
        self.planeStrainSigmaXX, self.planeStrainSigmaYY, self.planeStrainSigmaXY, self.planeStrainSigmaZZ = self.set_Stress_PlaneStrain()
        
        self.theta, self.PrincipalStrain, self.PrincipalStrainXX, self.PrincipalStrainYY = self.set_Principal_Strain()
        self.PrincipalStressXX, self.PrincipalStressYY, self.PrincipalStressZZ = self.set_Principal_Stress()
        self.HydrostaticStress = self.set_Hydrostatic_Stress()
        self.vonMisesStress = self.set_vonMises_Stress()
        self.eta = self.set_triaxiality()
        
        self.discfile = self.dislocationdir+disfile
        self.discdata = pd.read_csv(self.discfile)
        self.discX, self.discDens = self.set_discDensity()

        self.KASTM = self.calc_K_ASTM()
        self.PZsize_Analytical = self.calc_PlasticZoneSize_Analytical()        

        self.KStressFit, self.KStressFitLineX, self.KStressFitLineY = self.calc_K_StressFit()

        #self.PSsize_Measured = self.calc_PlasticZoneSize_Measured()

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
        ddXY = np.asmatrix(self.data["ddXY1"])
        
        epsXX = 10**6 * (dXX - self.d0)/self.d0
        epsYY = 10**6 * (dYY - self.d0)/self.d0
        epsXY = 10**6 * (dXY - self.d0)/self.d0
        depsXX = 10**6 * ddXX/self.d0
        depsYY = 10**6 * ddYY/self.d0
        depsXY = 10**6 * ddXY/self.d0

        return (XFromCrackTip, YFromCrackTip, epsXX,depsXX,epsYY,depsYY,epsXY,depsXY)
        
    def set_Stress_PlaneStrain(self):
    
        trueepsXX = 10**-6 * self.epsXX
        trueepsYY = 10**-6 * self.epsYY
        trueepsXY = 10**-6 * self.epsXY
        
        sigXX = (self.Youngs/((1-2*self.nu)*(1+self.nu)))*((1-self.nu)*trueepsXX+self.nu*trueepsYY)
        sigYY = (self.Youngs/((1-2*self.nu)*(1+self.nu)))*((1-self.nu)*trueepsYY+self.nu*trueepsXX)
        sigXY = (self.Youngs*trueepsXY)/(2*(1+self.nu))
        sigZZ = ((self.Youngs*self.nu)/((1-2*self.nu)*(1+self.nu)))*(trueepsYY+trueepsXX)
        
        return (sigXX, sigYY, sigXY, sigZZ)
        
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

        princStrain = []
        PrincipalStrainXX = []
        PrincipalStrainYY = []
        for i in range(len(np.asarray(self.epsXX)[0])):
            princStrain.append(RAR(np.asarray(theta)[0][i]).dot(np.array([np.array(self.epsXX)[0][i], np.array(self.epsYY)[0][i], np.array(self.epsXY)[0][i]])))
            PrincipalStrainXX.append(np.asarray(princStrain)[i][0][0])
            PrincipalStrainYY.append(np.asarray(princStrain)[i][0][1])
        return theta, princStrain, PrincipalStrainXX, PrincipalStrainYY

    def set_Principal_Stress(self):

        princStressXX = []
        princStressYY = []
        princStressZZ = []
        
        for i in range(len(np.asarray(self.PrincipalStrain))):
            strainXX = (10**-6) * np.asarray(self.PrincipalStrain)[i][0][0]
            strainYY = (10**-6) * np.asarray(self.PrincipalStrain)[i][0][1]
            princStressXX.append((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(strainXX*(1-self.nu) + strainYY*self.nu))
            princStressYY.append((self.Youngs/((1-2*self.nu)*(1+self.nu)))*(strainXX*(self.nu) + strainYY*(1-self.nu)))
            princStressZZ.append((self.Youngs/((1+self.nu))) * (self.nu/(1-2*self.nu)) * (strainXX + strainYY))
            
        return (princStressXX, princStressYY, princStressZZ)

    def set_Hydrostatic_Stress(self):
        return (1.0/3.0)*(np.asarray(self.PrincipalStressXX) + np.asarray(self.PrincipalStressYY) + np.asarray(self.PrincipalStressZZ))
              
    def set_vonMises_Stress(self):
        
        vonMisesStress = []
        
        for i in range(len(np.asarray(self.PrincipalStressXX))):
            s1 = self.PrincipalStressXX[i]
            s2 = self.PrincipalStressYY[i]
            s3 = self.PrincipalStressZZ[i]
            
            vonMisesStress.append(np.sqrt(0.5*((s1-s2)**2 + (s2-s3)**2 +(s1-s3)*3)))

        return vonMisesStress

    def set_triaxiality(self):
    
        eta = []
        
        for i in range(len(np.asarray(self.vonMisesStress))):
            eta.append(self.HydrostaticStress[i]/self.vonMisesStress[i])
        
        return eta
        
    def set_discDensity(self):
        
        XFromCrackTip = np.asmatrix(self.discdata["Xpos"])
        DislocationDensity = np.asmatrix(self.discdata["DiscDens"])
        return (XFromCrackTip, DislocationDensity)

    def calc_K_ASTM(self):

        return 0.03162*(self.load/(self.B*math.sqrt(self.W)))*((2+(self.length/self.W))/(1-(self.length/self.W))**(3/2))* \
               (0.866+4.64*(self.length/self.W)-13.32*(self.length/self.W)**2+14.72*(self.length/self.W)**3-5.6*(self.length/self.W)**4) #MPa sqrt(m)
        
    def calc_K_StressFit(self):
        
        linescan = [[self.X.tolist()[0][i], np.asarray(self.planeStrainSigmaYY)[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]

        X = [1/np.sqrt(i[0]) for i in linescan if i[0] > 0.1]

        Xfit = [1/np.sqrt(i[0]) for i in linescan if i[0] > self.PZsize_Analytical and i[0] < 1.5]
        Yfit = [i[1] for i in linescan if i[0] > self.PZsize_Analytical and i[0] < 1.5]
       
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Xfit,Yfit)
        line = slope*np.asarray(X) + intercept
        KStressFit = slope*np.sqrt(2*np.pi/1000)
        
        return KStressFit, X, line

    def calc_PlasticZoneSize_Analytical(self):
        return (1/(3*np.pi))*1000*(self.KASTM/self.YieldStrength)**2  #mm

    def PlotStressVsSqrtR(self):
        def tick_function(x):
            V = (1/x**2)
            return ["%.2f" % z if (z<0.1) else "%.1f" % z if (z<1) else "%.0f" % z for z in V]

    
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        
        color = colors[0]
        marker = markers[0]

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_sYYvsSqrtr"
        
        linescan = [[self.X.tolist()[0][i], np.asarray(self.planeStrainSigmaYY)[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [1/np.sqrt(i[0]) for i in linescan if i[0] > 0.0]
        Y = [i[1] for i in linescan if i[0] > 0.0]

###############################################################
## Plot Data
# Primary Axis
        ax1.scatter(X,Y,color=color,marker=marker)
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
        ax1.plot(self.KStressFitLineX,self.KStressFitLineY)
        
###############################################################
## Plot Plastic Zone Boundary (Analytical)

        ax1.axvline(x=1/np.sqrt(self.PZsize_Analytical), color=colors[0], linestyle='dashed')
        ax1.text(1/np.sqrt(self.PZsize_Analytical) + 0.15,800, r'$\mathrm{r}_p$',fontsize=18) 
        # plt.show()

        fig.savefig("../Acta Material Paper/Images/Stresses/"+fileLabel+".pdf",bbox_inches='tight')
        plt.close(fig)

    def PlotAllStrains(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_Strain"

        for e in [self.epsXX, self.epsYY, self.epsXY]:
            if np.array_equal(e,self.epsXX):
                label = r'$\epsilon\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.epsYY):
                    label = r'$\epsilon\prime_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                    
                else:
                    if np.array_equal(e,self.epsXY):
                        label = r'$\epsilon\prime_{XY}$'
                        color = colors[2]
                        marker = markers[2]                    

    #####
    ## Plot strain
    #####   
 
            linescan = [[self.X.tolist()[0][i], e.tolist()[0][i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]

            ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
            ax1.legend(loc='upper right')
            ax1.set_title("Strain")    
            ax1.set_xlabel('Distance from crack tip (mm)')
            ax1.set_ylabel(r'Strain / $10^-6$')
            ax1.axis([0,12,-500,3000])
            for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                         ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
                
#        plt.show()
        fig.savefig("../Acta Material Paper/Images/Strains/"+fileLabel+".pdf",bbox_inches='tight')
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
                
#        plt.show()
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

    def PlotAllPrincipalStrains(self):
    
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        fileLabel = self.env + "_" + str(self.load) + "_a" + str(round(self.length,2)) + "_K" + str(round(self.KASTM,2))+"_PrincipalStrain"

        for e in [self.PrincipalStrainXX, self.PrincipalStrainYY]:
            if np.array_equal(e,self.PrincipalStrainXX):
                label = r'$\epsilon\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.PrincipalStrainYY):
                    label = r'$\epsilon\prime_{YY}$'
                    color = colors[1]
                    marker = markers[1]

    #####
    ## Plot strain
    #####   
 
            linescan = [[self.X.tolist()[0][i], e[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]

            ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
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
                label = r'$\sigma\prime_{XX}$'
                color = colors[0]
                marker = markers[0]
            else:
                if np.array_equal(e,self.PrincipalStressYY):
                    label = r'$\sigma\prime_{YY}$'
                    color = colors[1]
                    marker = markers[1]
                else:
                    if np.array_equal(e,self.PrincipalStressZZ):
                        label = r'$\sigma\prime_{ZZ}$'
                        color = colors[2]
                        marker = markers[2]
    #####
    ## Plot strain
    #####        
            linescan = [[self.X.tolist()[0][i], e[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
            X = [i[0] for i in linescan]
            Y = [i[1] for i in linescan]

            ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
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

        label = r'$\sigma\prime_{VM}$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot vonMisesStress
    #####        
        linescan = [[self.X.tolist()[0][i], self.vonMisesStress[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]

        ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
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

        label = r'$\sigma_H$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot hydrostatic stress
    #####
    
        linescan = [[self.X.tolist()[0][i], self.HydrostaticStress[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]

        ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
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

        label = r'$\eta$'
        color = colors[0]
        marker = markers[0]

    #####
    ## Plot triaxiality
    #####        
        linescan = [[self.X.tolist()[0][i], self.eta[i]] for i,j in enumerate(self.Y.tolist()[0]) if j > -0.01 and j < 0.01]
        X = [i[0] for i in linescan]
        Y = [i[1] for i in linescan]

        ax1.plot(X,Y,color=color,linestyle='-',marker=marker,label=label)
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
    for hc in H2cracks:
        X.append(hc.length)
        Y.append(hc.KStressFit)

    label = "Hydrogen"
    color = colors[0]
    marker = markers[0]
    ax1.plot(X,Y,color=color,linestyle='None',marker=marker,label=label)

## Plot Air Data
    X = []
    Y = []
    for ac in aircracks:
        X.append(ac.length)
        Y.append(ac.KStressFit)

    label = "Air"
    color = colors[1]
    marker = markers[1]
    ax1.plot(X,Y,color=color,linestyle='None',marker=marker,label=label)
    
    ax1.set_title(r'K vs Crack Length')    
    ax1.set_xlabel('Crack Length (mm)')
    ax1.set_ylabel(r'K$_{app}$ (MPa m $^{1/2}$)')

    ax1.axis([5,20,0,50])
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)
            
    # plt.show()
    fig.savefig("../Acta Material Paper/Images/"+fileLabel+".pdf",bbox_inches='tight')
    plt.close(fig)

## Function for writing all data to .xlsx spreadsheet
def writeDatatoXLSX(cracklist):
    #Turn 
###################################################################################################
################################## Main program ###################################################
###################################################################################################
       
#Create crack instances
B = 3.0
W = 26.67
crackList = []

######Hydrogen Cracks #####
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':3.275, 'crackPosY':0.075, 'dataFile':"H2_Crack1_Load850_1179_1425_fit_5_22-Dec-2017.txt", 'discfile':"Crack1_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700,'crackPosX':3.275, 'crackPosY':0.15, 'dataFile':"H2_Crack1_Load1700_931_1177_fit_5_22-Dec-2017.txt", 'discfile':"Crack1_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':3.405, 'crackPosY':-0.015, 'dataFile':"H2_Crack2_Load850_1675_1921_fit_5_22-Dec-2017.txt", 'discfile':"Crack2_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':3.405, 'crackPosY':-0.080, 'dataFile':"H2_Crack2_Load1700_1427_1673_fit_5_22-Dec-2017.txt", 'discfile':"Crack2_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':3.80, 'crackPosY':0.015, 'dataFile':"H2_Crack3_Load850_2171_2417_fit_5_22-Dec-2017.txt", 'discfile':"Crack3_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':3.80, 'crackPosY':-0.055, 'dataFile':"H2_Crack3_Load1700_1923_2169_fit_5_22-Dec-2017.txt", 'discfile':"Crack3_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':4.25, 'crackPosY':0.025, 'dataFile':"H2_Crack4_Load850_2667_2913_fit_5_22-Dec-2017.txt", 'discfile':"Crack4_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':4.25, 'crackPosY':-0.04, 'dataFile':"H2_Crack4_Load1700_2419_2665_fit_5_22-Dec-2017.txt", 'discfile':"Crack4_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':5.3, 'crackPosY':0.05, 'dataFile':"H2_Crack5_Load850_3163_3409_fit_5_22-Dec-2017.txt", 'discfile':"Crack5_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':5.3, 'crackPosY':-0.039, 'dataFile':"H2_Crack5_Load1700_2915_3161_fit_5_22-Dec-2017.txt", 'discfile':"Crack5_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':6.185, 'crackPosY':0.035, 'dataFile':"H2_Crack6_Load850_3660_3906_fit_5_22-Dec-2017.txt", 'discfile':"Crack6_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':6.185, 'crackPosY':-0.042, 'dataFile':"H2_Crack6_Load1700_3412_3658_fit_5_22-Dec-2017.txt", 'discfile':"Crack6_Load1700.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':6.60, 'crackPosY':0.045, 'dataFile':"H2_Crack7_Load850_4156_4402_fit_5_20-Dec-2017.txt", 'discfile':"Crack7_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':6.60, 'crackPosY':0.048, 'dataFile':"H2_Crack7_Load1700_3908_4154_fit_5_05-Feb-2018.txt", 'discfile':"Crack7_Load1700.csv"}) 
# crackList.append({'Condition':"Hydrogen", 'Load':850, 'crackPosX':7.03, 'crackPosY':0.05, 'dataFile':"H2_Crack7_Load850_4156_4402_fit_5_20-Dec-2017.txt", 'discfile':"Crack8_Load850.csv"})
# crackList.append({'Condition':"Hydrogen", 'Load':1700, 'crackPosX':7.03, 'crackPosY':-0.04, 'dataFile':"H2_Crack6_Load1700_3412_3658_fit_5_22-Dec-2017.txt", 'discfile':"Crack8_Load1700.csv"})

#####Air Cracks #####
crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':4.3, 'crackPosY':0.1, 'dataFile':"Air_Crack1_Load850_5898_6144_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack1_Load850.csv"})
# crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':4.3, 'crackPosY':0.04, 'dataFile':"Air_Crack1_Load1700_5650_5896_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack1_Load1700.csv"})
# crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':5.12, 'crackPosY':0.135, 'dataFile':"Air_Crack2_Load850_6394_6640_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack2_Load850.csv"})
# crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':5.12, 'crackPosY':0.1, 'dataFile':"Air_Crack2_Load1700_6146_6392_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack2_Load1700.csv"})
# crackList.append({'Condition':"Air", 'Load':850, 'crackPosX':6.36, 'crackPosY':0.15, 'dataFile':"Air_Crack3_Load850_6890_7136_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack3_Load850.csv"})
# crackList.append({'Condition':"Air", 'Load':1700, 'crackPosX':6.36, 'crackPosY':0.08, 'dataFile':"Air_Crack3_Load1700_6642_6888_fit_5_03-Jan-2018.txt", 'discfile':"Air_Crack3_Load1700.csv"})

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

PmaxCracks = [c for c in crack if c.load == 1700]

for i,c in enumerate(crack):
    # print i
    # c.PlotStressVsSqrtR()
    # c.PlotAllStrains()
    # c.PlotAllPrincipalStrains()
    # c.PlotAllPrincipalStresses()
    # c.PlotvonMisesStress()
    # c.PlotTriaxiality()
    # c.PlotTriaxialityContour()
    # c.PlotvonMisesStressContour()
    # c.PlotAllStrainsContour()
    # c.PlotHydrostaticStress()
    # c.PlotDislocationDensity()
    # c.PlotAllStresses()
# PlotKvsa(PmaxCracks)
