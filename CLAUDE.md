# Welcome to the Ruckus repository! Here are a few things to know: 

# Environment
- The project is in a conda environment. You can invoke the environment with the `source ~/.bash_profile && conda activate ruckus` command and run pip commands, tests, etc. 
- The project is separated into two different major components "ruckus-server" and "ruckus-agent", with a commons directory that serves as the middle repository. 

# Development
- Generally, you should prefer a test-driven-development pattern when feasible. NEVER continue onto another feature without ensuring the tests pass and NEVER skip tests. We need to ensure 100% test passing before moving on and removing or commenting out tests is unacceptable. When finished with the test, prompt the user with a summary of the tests and what correctness properties they are asserting and whether they are using mock data vs. real endpoints. Once the user has approved the tests, run the tests to confirm they fail, then commit the tests before starting development.  
- When prompted with a feature request, examine the entire repository to understand which files will be relevant, ultrathink to come up with a comprehensive plan, then ask any clarifying or necessarily design questions to the user before continuing. Never write code without a comprehensive design review first. 
- We are in a git environment. When finishing a feature, add the relevant files, write a comprehensive commit message, and prompt the user to push the changes manually. 

# Testing
- Prefer to run single tests, not the entire test suite, for performance, except when verifying completion and submission of a feature.
- Be sure to typecheck when you're done making a series of code changes.  
