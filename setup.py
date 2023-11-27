from setuptools import setup

package_name = 'uav_navigation'

setup(
    name=package_name,
    version='0.1.0',    
    description='Code repository to perform reinforcement learning experiments for navigation using the gym-webots-drone environment',
    url='https://github.com/angel-ayala/uav-navigation',
    author='Angel Ayala',
    author_email='aaam@ecomp.poli.br',
    license='GPL-3.0',
    packages=[package_name],
    install_requires=['gym==0.26.0''],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Games/Entertainment :: Simulation',
    ],
)
