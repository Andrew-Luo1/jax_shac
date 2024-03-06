""" 
Available XML Files
"""
from pathlib import Path
slider_motor_xml = str(Path(Path(__file__).parent, Path("shac_test"), Path("slider_motor.xml")))
vslider_motor_xml = str(Path(Path(__file__).parent, Path("shac_test"), Path("slider_motor_vision.xml")))
slider_position_xml = str(Path(Path(__file__).parent, Path("shac_test"), Path("slider_position.xml")))
anymal_xml = str(Path(Path(__file__).parent, Path("assets/anybotics_anymal_c"), Path("anymal_c_torque.xml")))