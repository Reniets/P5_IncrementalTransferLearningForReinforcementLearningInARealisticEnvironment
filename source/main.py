import source.manual_control as manualControl
from source.carla_environment import CarlaEnvironment
import source.data_handler as dataHandler


def runCarlaServer(makeVideo=True, port="2000", runTestEnvironment=True, runManualControl=False):
    if makeVideo: dataHandler.clearFrameFolder()  # Clear so no previous frames ruin the video

    carlaEnv = CarlaEnvironment(port)

    if runTestEnvironment: carlaEnv.setupTestEnvironment()
    else: carlaEnv.create()

    if runManualControl: manualControl.main()

    # Wait for user to terminate simulation
    input("Press Enter to close...")
    carlaEnv.close()

    if makeVideo: dataHandler.makeVideoFromSensorFrames()  # Concat all images in data/frames into a video


runCarlaServer()
