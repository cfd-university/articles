This repository contains the raw HTML of all articles hosted on [cfd.university](https://cfd.university/)

Having all articles on GitHub allows for collaborative improvements of articles. If you have found issues, inconsistencies, or errors with the articles, please feel free to open up a new issue.

The articles are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Contribution guideline

Depending of the nature of your contribution, you may choose to do one of two things:

1. Submit a pull request for small changes (e.g. typos, syntax errors, etc.)
2. Open an issue for anything more substantial (e.g. logical errors, changes requiring rewriting or expansion of articles)

Once an issue has been closed and integrated, or a pull request has been accepted and merged, your changes should propagate to the main website.

> [!NOTE]
> This feature is currently under implementation

The HTML files follow mostly plain vanilla HTML with some additional enhancements made by WordPress. You can preview any changes using the [online Gutenberg](https://wordpress.org/gutenberg/) editor. To do that, follow these steps:

1. Use the key combination ```Ctrl+Alt+Shift+M``` to change from visual to HTML mode.
2. Select all text and delete it.
3. Copy the changes you made on the HTML and paste them here.
4. Change back into visual mode using the same key combination (```Ctrl+Alt+Shift+M```)

If you copy the entire article, you may see that some parts will not display properly (things like equations and code snippets). These use third-party plugins and will not render properly in the online editor linked above. If in doubt, open a issue rather than submitting a pull request.

If you make changes to the text directly, please note the following:

- Do not use more than 550 characters per paragraph (about 90 words)
- All equations use the LaTeX equation syntax and the online rendered [katex](https://katex.org/). You can go to this website and check that any equation you type can be represented by katex.
- You cannot make changes directly to code sections, as syntax highlighting has to be manually generated (this is currently also being worked on, but as long as we work with static HTML files, we cannot change that). Please open an issue for any code related changes.
- Section headings should also not be tampered with. Fixing typos is fine, but anything more and the internal linking process may break. Just like the code section, I do generate these manually on the backend and copy and paste them into the HTML. Any changes here should also be requested through an issue on GitHub.



