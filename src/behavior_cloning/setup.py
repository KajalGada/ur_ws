from setuptools import setup

package_name = 'behavior_cloning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='UR5 behavior cloning inference node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'infer_node = behavior_cloning.infer_node:main',
        ],
    },
)