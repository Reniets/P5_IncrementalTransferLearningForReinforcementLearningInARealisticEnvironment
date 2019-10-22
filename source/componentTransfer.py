import json
import shutil
import zipfile
import os
from collections import OrderedDict
import io
import numpy as np


class ComponentTransfer:
    def __init__(self):
        self.fromAgentSavePath = None
        self.toAgentSavePath = None
        self.transferAgentSavePath = None
        self.parameterNamesToTransfer = None
        self.parametersToTransfer = {}

    def transfer(self, fromAgentSavePath, toAgentSavePath, parameterIndicesToTransfer):
        # Setup
        self.fromAgentSavePath = fromAgentSavePath
        self.toAgentSavePath = toAgentSavePath
        self.transferAgentSavePath = 'Transfer_FromLevel_' + fromAgentSavePath.split('_')[2] + '_ToLevel_' + toAgentSavePath.split('_')[2]
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
            with zipfile.ZipFile(agentFromZip.open('parameters')) as parametersZip:
                for parameter in self.parameterNamesToTransfer:
                    with parametersZip.open(parameter + '.npy') as parameterNpy:
                        self.parametersToTransfer[parameter] = np.load(parameterNpy)

    def _toAgentExtract(self):
        with zipfile.ZipFile(self.toAgentSavePath) as agentToZip:
            agentToZip.extractall('ToAgentExtracted')

    def _loadToAgentParams(self):
        npzFileParameters = np.load('ToAgentExtracted/parameters')
        toAgentParams = OrderedDict(npzFileParameters)

        del npzFileParameters  # Delete object to allow removing ToAgentExtracted directory later

        return toAgentParams

    def _paramsToBytes(self, params):
        # Create byte-buffer and save params with
        # savez function, and return the bytes.
        byteFile = io.BytesIO()
        np.savez(byteFile, **params)
        serializedParams = byteFile.getvalue()
        return serializedParams

    def _isJsonSerializable(self, item):
        # Try with try-except struct.
        jsonSerializable = True
        try:
            _ = json.dumps(item)
        except TypeError:
            jsonSerializable = False
        return jsonSerializable

    def _dataToJson(self, data):
        # First, check what elements can not be JSONfied,
        # and turn them into byte-strings
        serializableData = {}
        for dataKey, dataItem in data.items():
            # See if object is JSON serializable
            if self._isJsonSerializable(dataItem):
                # All good, store as it is
                serializableData[dataKey] = dataItem
            else:
                raise Exception("Error")

        jsonString = json.dumps(serializableData, indent=4)
        return jsonString

    def _serializeAndExportTransferAgent(self, toAgentParams):
        with open('ToAgentExtracted/data') as jsonFile:
            data = json.load(jsonFile)

        serializedData = self._dataToJson(data)
        serializedParams = self._paramsToBytes(toAgentParams)

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


