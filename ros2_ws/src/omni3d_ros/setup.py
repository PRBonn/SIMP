import os
from glob import glob
from setuptools import setup

package_name = 'omni3d_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch')),
        (os.path.join('share', package_name), glob('configs/*')),
        (os.path.join('share', package_name), glob('trained_models/*')),
        (os.path.join('lib', package_name), glob('omni3d_ros/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nickybones',
    maintainer_email='nickyfullmetal@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'GTNode = omni3d_ros.GTNode:main',
        'SIMPNode = omni3d_ros.SIMPNode:main',
        'Omni3DNode = omni3d_ros.Omni3DNode:main',
        'Omni3DMappingNode = omni3d_ros.Omni3DMappingNode:main',
        ],
    },
)


