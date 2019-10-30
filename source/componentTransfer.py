import json
import shutil
import zipfile
import os
from collections import OrderedDict
import io
import numpy as np
import sys
from os import path
sys.path.append(path.abspath('../../stable-baselines'))
from stable_baselines.common.save_util import params_to_bytes, data_to_json


class ComponentTransfer:
    def __init__(self):
        self.fromAgentSavePath = None
        self.toAgentSavePath = None
        self.transferAgentSavePath = None
        self.parameterNamesToTransfer = None
        self.parametersToTransfer = {}

    def transfer(self, fromAgentSavePath, toAgentSavePath, toLevel,  parameterIndicesToTransfer):
        # Setup
        self.fromAgentSavePath = fromAgentSavePath
        self.toAgentSavePath = toAgentSavePath
        self.transferAgentSavePath = 'TransferAgentLogs/Transfer_FromLevel_' + fromAgentSavePath.split('_')[2] + '_ToLevel_' + str(toLevel)
        self.parameterNamesToTransfer = self._getParametersToTransfer(parameterIndicesToTransfer)

        self._loadParametersToTransfer()
        self._toAgentExtract()

        toAgentParams = self._loadToAgentParams()

        # Perform transfer of parameters
        for parameter in self.parameterNamesToTransfer:
            toAgentParams[parameter] = self.parametersToTransfer[parameter]

        self._serializeAndExportTransferAgent(toAgentParams)

        # Clean extracted folder
        shutil.rmtree('ToAgentExtracted')
        os.remove('parameters')

    def _getParametersToTransfer(self, parameterIndicesToTransfer):
        parameterList = [
            "model/c1/w:0",
            "model/c1/b:0",
            "model/c2/w:0",
            "model/c2/b:0",
            "model/c3/w:0",
            "model/c3/b:0",
            "model/fc1/w:0",
            "model/fc1/b:0",
            "model/vf/w:0",
            "model/vf/b:0",
            "model/pi/w:0",
            "model/pi/b:0",
            "model/q/w:0",
            "model/q/b:0"
        ]

        return [parameterList[i] for i in parameterIndicesToTransfer]

    def _loadParametersToTransfer(self):
        with zipfile.ZipFile(self.fromAgentSavePath) as agentFromZip:
            agentFromZip.extract('parameters')

        self.parametersToTransfer = np.load('parameters')

    def _toAgentExtract(self):
        with zipfile.ZipFile(self.toAgentSavePath) as agentToZip:
            agentToZip.extractall('ToAgentExtracted')

    def _loadToAgentParams(self):
        npzFileParameters = np.load('ToAgentExtracted/parameters')
        toAgentParams = OrderedDict(npzFileParameters)

        del npzFileParameters  # Delete object to allow removing ToAgentExtracted directory later

        return toAgentParams

    def _serializeAndExportTransferAgent(self, toAgentParams):
        with open('ToAgentExtracted/data') as jsonFile:
            data = json.load(jsonFile)

        serializedData = data_to_json(data)
        serializedParams = params_to_bytes(toAgentParams)

        serializedParamList = json.dumps(
            list(toAgentParams.keys()),
            indent=4
        )

        # Add zip to save path
        if isinstance(self.transferAgentSavePath, str):
            _, ext = os.path.splitext(self.transferAgentSavePath)
            if ext == "":
                self.transferAgentSavePath += ".zip"

        # Serialize data and write to zip file
        with zipfile.ZipFile(self.transferAgentSavePath, "w") as file_:
            if serializedData is not None:
                file_.writestr("data", serializedData)
            if toAgentParams is not None:
                file_.writestr("parameters", serializedParams)
                file_.writestr("parameter_list", serializedParamList)


