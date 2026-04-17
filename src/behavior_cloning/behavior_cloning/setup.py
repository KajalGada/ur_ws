from setuptools import setup

package_name = 'behavior_cloning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Behavior cloning package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'behavior_cloning_infer_node = behavior_cloning.infer_node:main',
        ],
    },
)