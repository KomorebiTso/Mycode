# -*- coding: UTF-8 -*-
# -*- coding: gbk -*-

# -*- coding: cp936 -*-
# -*- coding: mbcs -*-


from abaqus import *
from abaqusConstants import *
from caeModules import *
from viewerModules import *
from driverUtils import executeOnCaeStartup
import odbAccess
import regionToolset

import numpy as np 
import sympy as sp
import scipy.fftpack

def gaussian_random_field(alpha,size,mean,std,flag_normalize=True):
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_idx = scipy.fftpack.fftshift(k_ind)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 4.0)
    amplitude[0, 0] = 0
    noise = np.random.normal(size=(size, size)) + 1j * np.random.normal(size=(size, size))
    gfield = np.fft.ifft2(noise * amplitude).real
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield / np.std(gfield)

    return gfield * std + mean


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

modelName='Model-1'
partName='Part-1'
searchR = 0.002
numCpus = 10
runNum = 1

session.viewports['Viewport: 1'].assemblyDisplay.predefinedFieldOptions.setValues(temperatureField=OFF)
model = mdb.models[modelName]
asm = model.rootAssembly
part=model.parts[partName]
pNode = part.nodes
inst = asm.instances['Part-1-1']
instNode = inst.nodes


for runC in range(runNum):
    for pffn in model.predefinedFields.keys():
        del model.predefinedFields[pffn]

    nodeOldLabel = []
    nodeNewLabel = [n.label for n in pNode]

    tempArray = gaussian_random_field(5.,201,25,20)
    rowNum = len(tempArray)
    colNum = len(tempArray[0])
    crdI = 0.01
    crdTmpDict = {(crdI*k2, crdI*k1):tempArray[k1,k2] for k1 in range(rowNum) for k2 in range(colNum)}

    for idx,crd in enumerate(crdTmpDict.keys()):
        if idx+1==1 or (idx+1)%1000==0 or idx+1==len(crdTmpDict.keys()):
            print('[Info] Ing . . .  %s/%s'%(idx+1,len(crdTmpDict.keys())))
        nodeTemp = pNode.getByBoundingSphere(center=(crd[0],crd[1],0.0),radius=searchR)
        if len(nodeTemp)<=0:
            continue

        region = asm.Set(name='PFD-TMP-%s'%(idx+1), nodes=instNode.sequenceFromLabels(labels=[n.label for n in nodeTemp if n.label not in nodeOldLabel]))
        temp = crdTmpDict[crd]
        model.Temperature(name='Predefined Field-%s'%(idx+1), 
            createStepName='Initial', region=region, distributionType=UNIFORM, 
            crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(temp, ))
        for n in nodeTemp:
            nodeOldLabel.append(n.label)
            nodeNewLabel.remove(n.label)

    for nl in nodeNewLabel:
        nseq = instNode.sequenceFromLabels(labels=[nl,])
        node = nseq[0]
        temp = crdTmpDict[(round(node.coordinates[0]/crdI)*crdI, round(node.coordinates[1]/crdI)*crdI)]
        region = asm.Set(name='PFD-TMP-%s'%(idx+1), nodes=nseq)
        model.Temperature(name='Predefined Field-%s'%(idx+1), 
            createStepName='Initial', region=region, distributionType=UNIFORM, 
            crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(temp, ))

    #session.viewports['Viewport: 1'].assemblyDisplay.predefinedFieldOptions.setValues(temperatureField=ON)
    mdb.save()

    jobName = 'Job-%s'%runC
    mdb.Job(name=jobName, model=modelName, description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=numCpus, 
        numDomains=numCpus, numGPUs=0)

    mdb.jobs[jobName].submit()
    mdb.jobs[jobName].waitForCompletion()