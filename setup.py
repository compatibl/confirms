import setuptools

with open('./README.md', 'r') as readme_file:
    readme = readme_file.read()

with open('./install_requires.txt') as install_requires_file:
    install_requires = [line.strip() for line in install_requires_file.readlines()]

setuptools.setup(
    name='confirms',
    version='0.0.1',
    author='The Project Contributors',
    description='Model Governance AI for Trade Confirmations',
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    url='https://github.com/compatibl/confirms',
    project_urls={
        'Source Code': 'https://github.com/compatibl/confirms',
    },
    packages=setuptools.find_packages(include=('confirms', 'confirms.*'), exclude=('tests', 'tests.*')),
    classifiers=[
        # Alpha - will attempt to avoid breaking changes but they remain possible
        'Development Status :: 3 - Alpha',

        # Audience and topic
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',

        # License
        'License :: OSI Approved :: Apache Software License',

        # Runs on Python 3.9 and later releases
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',

        # Operating system
        'Operating System :: OS Independent',
    ],
)
