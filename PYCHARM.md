# Use PyCharm to Run Code

While the Langchain package now lists Python 3.14 as compatible, there are some compatibility issues, and it's recommended that you use Python 3.13 for this project.

## Install Python 

1. Install Version 3.13.  

If you haven't installed it yet, download the installer for your operating system from the official [https://www.python.org/downloads/](Python downloads page).   

Windows Users: During installation, ensure you check the box that says "Add python.exe to PATH"

2. Add the Interpreter to PyCharm. 

Once installed, tell PyCharm to use this specific version for your project:  

1.  Open your project in PyCharm.  
2.  Go to File | Settings (Windows/Linux) or PyCharm | Settings (macOS).  
3.  Navigate to Project: [Your Project Name] | Python Interpreter.  
4.  Click the Add Interpreter link (often a small drop-down or plus icon) and select Add Local Interpreter....    
5.  In the left sidebar, choose Virtualenv Environment (recommended for GitHub projects).  
6.  In the Base interpreter field, click the three dots ... to browse.  
7.  Find and apply the Python 3.13 executable:  
    - `which python3.13` (MAC)
    - `where python3.13` (WINDOWS path search)
    - `find`, `locate`, or `whereis` (LINUX)  
8.  Click OK to save.  

PyCharm will now create a new virtual environment using Python 3.13 for this project.  

## 3. Verify the Version. 
To confirm it worked, look at the bottom-right corner of the PyCharm window; it should display Python 3.13. Alternatively, open the Terminal tab at the bottom and type:python --version


## Start Project
To download a project from GitHub and set it up in PyCharm, you can "clone" the repository directly within the IDE. This process downloads all files and automatically initializes them as a new project.

## Step 1: Copy the Repository URL
1. Navigate to the project's main page on GitHub.  
2. Click the green Code button.
3. Copy the URL under the HTTPS tab. 

## Step 2: Clone into PyCharm. 
The exact menu option depends on whether you have a project already open:  
- From the Welcome Screen: Click Get from VCS.  
- If a Project is Already Open: Go to File | New | Project from Version Control or Git | Clone (in newer versions).  

In the dialog that appears:  

1. Paste the URL into the URL field.  
2.  Choose a Directory on your computer where the files will be saved.  
3.  Click Clone.  

##Step 3: Configure the Python Interpreter. 
Once the files are downloaded, PyCharm may prompt you to create a virtual environment if it detects a requirements.txt file. If it does not, you must set one up manually to run the code:  

1. Go to File | Settings (Windows/Linux) or PyCharm | Settings (macOS).  
2. Select Project: [Your Project Name] | Python Interpreter.  
3. Click Add Interpreter and follow the prompts to create a new environment.    

After the Python Interpreter is set, you can run files by right-clicking them in the project tree and selecting Run.

## Install Dependencies

Once you have your Python 3.13 interpreter set up, you need to install the project’s dependencies (libraries like Langchain, Pydantic, etc.). These dependencies are listed in a file named `requirements.txt`.  

### Option 1: Automatic Detection (Recommended). 
PyCharm often detects these files automatically when you open a project or a Python file.  
1. Look for a yellow notification bar at the top of the editor that says "Package requirements are not satisfied".  
2. Click the Install requirements link within that bar.  
3.  PyCharm will handle the pip install process for you.  

### Option 2: Using the Terminal. 
If the notification doesn't appear, you can manually trigger the installation using the built-in PyCharm Terminal:  
1. Open the Terminal tab at the bottom of PyCharm (or press Alt + F12).   
2. The terminal should automatically activate your project's virtual environment (indicated by (venv) at the start of the line).  
3. Run the following command:pip install -r requirements.txt. 
