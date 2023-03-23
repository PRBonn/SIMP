import os
from glob import glob
from setuptools import setup

package_name = 'omni3d_ros'

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name), glob('omni3d_ros/*')),
    ]

def package_files(data_files, directory_list, dest_list):

    paths_dict = {}

    for i, directory in enumerate(directory_list):

        for (path, directories, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join(dest_list[i], package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)

                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=package_files(data_files, ['models/', 'launch/', 'configs/', 'omni3d/' ], ['share', 'share', 'share', 'lib', 'lib' ]),

    #     (os.path.join('share', package_name), glob('launch/*.launch')),
    #     (os.path.join('share', package_name), glob('configs/*')),
    #     (os.path.join('share', package_name), glob('trained_models/*')),
    #     (os.path.join('lib', package_name), glob('omni3d_ros/*'))
    #     (os.path.join('lib', package_name), glob('omni3d/*'))
    # ],
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


