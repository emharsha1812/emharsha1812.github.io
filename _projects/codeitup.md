---
layout: page
title: CodeItUp
description: A versatile online code editor with multi-language support
img: assets/img/codeitup.png
importance: 1
category: work
---

# CodeItUp: Your Online Coding Companion

[CodeItUp](https://reactcodeeditor.netlify.app/) is a powerful and user-friendly online code editor designed to streamline your coding experience. Built with React and enhanced with TailwindCSS, this project offers a seamless environment for writing, compiling, and executing code in over 40 programming languages.

<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/codeitup-main.png" title="CodeItUp Main Interface" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/codeitup-theme.png" title="Theme Selection" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: The main interface of CodeItUp showcasing the code editor and output panel. Right: A glimpse of the theme selection feature, allowing users to customize their coding environment.
</div>

## Key Features

- **Multi-Language Support**: Write and execute code in 40+ programming languages, catering to diverse development needs.
- **Theme Customization**: Personalize your coding environment with a variety of available themes to suit your preferences.
- **Instant Compilation**: Leverage the power of Judge0API to compile and run your code seamlessly within the browser.

## Technical Implementation

CodeItUp is built using modern web technologies to ensure a robust and responsive user experience:

- **Frontend**: Developed with Create React App, providing a fast and efficient single-page application.
- **Styling**: Utilizes TailwindCSS for a sleek, responsive design that looks great on all devices.
- **Code Execution**: Integrates with Judge0API to handle code compilation and execution securely.

<div class="row justify-content-center">
    <div class="col-8 col-md-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/codeitup-execution.png" title="Code Execution Example" class="img-fluid rounded z-depth-1 smaller-image" %}
    </div>
</div>
<div class="caption">
    An example of code execution in CodeItUp, demonstrating the output panel and error handling capabilities.
</div>

## Future Enhancements

While CodeItUp is already a powerful tool, there are exciting plans for future improvements:

1. User authentication and registration using Firebase Auth
2. Personalized user profile pages
3. Code saving functionality with Firestore integration
4. Code sharing capabilities for collaborative coding

## Try It Out

Experience CodeItUp for yourself at [https://reactcodeeditor.netlify.app/](https://reactcodeeditor.netlify.app/). Whether you're a seasoned developer or just starting your coding journey, CodeItUp provides the tools you need to write, test, and perfect your code in a user-friendly online environment.

For developers interested in contributing or running the project locally, check out the [GitHub repository](https://github.com/emharsha1812/CodeitUp) for installation instructions and more details.