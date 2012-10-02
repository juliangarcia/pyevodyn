import os
from setuptools import setup
#from setuptools import find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
if __name__ == '__main__':        
    setup(
        name = "pyevodyn",
        version = "0.0.1",
        # Package structure
        #
        # find_packages searches through a set of directories 
        # looking for packages
        packages=['pyevodyn'],
        #packages = find_packages(exclude = ['*.tests', '*.tests.*', 'tests.*', 'tests']),
        
        # package_dir directive maps package names to directories.
        # package_name:package_directory
        #THIS LINE HAS BEEN COMMENTED TO GET RID OF SRC
        #package_dir = {'': 'src'},
        
        # Dependencies
        #
        # Dependency expressions have a package name on the left-hand 
        # side, a version on the right-hand side, and a comparison 
        # operator between them, e.g. == exact version, >= this version
        # or higher
        install_requires = [
            'numpy>=1.6.2 ',
            'pandas>=0.8.0',
            'sympy>=0.7.1'
        ],
          
        # Tests
        #
        # Tests must be wrapped in a unittest test suite by either a
        # function, a TestCase class or method, or a module or package
        # containing TestCase classes. If the named suite is a package,
        # any submodules and subpackages are recursively added to the
        # overall test suite.
        #test_suite = 'greatings.tests.suite',
        # Download dependencies in the current directory
        #tests_require = 'docutils >= 0.6',
        
        # Meta information
        author = "Julian Garcia",
        author_email = "garcia@evolbio.mpg.de",
        description = ("PyEvoDyn: A tool to study and teach evolutionary dynamics using python."),
        license = "BSD",
        keywords = "evolution dynamics complex systems",
        url = "http://garciajulian.com",
        long_description=read('README'),
        classifiers=[
            "Development Status :: 2 - Pre Alpha",
            "Topic :: Utilities",
            "License :: OSI Approved :: BSD License",
        ],
        zip_safe=True      
    )