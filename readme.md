This repository contains the raw HTML of all articles hosted on [cfd.university](https://cfd.university/)

Having all articles on GitHub allows for collaborative improvements of articles. If you have found issues, inconsistencies, or errors with the articles, please feel free to open up a new issue.

The articles are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

> [!NOTE] 
> This repository is currently incomplete and its integration with the main website is ongoing. 

## Contribution guide

If you feel more comfortable with git and pull requests in particular, you can also make direct changes to the raw HTML and submit a pull request. Once a pull request has been accepted and merged with this repository, your changes should automatically propagate to the main website.

The HTML files follow mostly plain vanilla HTML with some additional enhancements made by WordPress. If you are just submitting a pull request for simple changes to the text (editing paragraphs), you should be able to find your way around. If you want to make more substantial changes, you can preview any changes using the [online Gutenberg](https://wordpress.org/gutenberg/) editor. To do that, follow these steps:

1. Use the key combination ```Ctrl+Alt+Shift+M``` to change from visual to HTML mode.
2. Select all text and delete it.
3. Copy the changes you made on the HTML and paste them here.
4. Change back into visual mode using the same key combination (```Ctrl+Alt+Shift+M```)

If you copy the entire article, you may see that some parts will not display properly (things like equations and code snippets). These use third-party plugins and will not render properly in the online editor linked above.

### Writing style guide

When editing text, please make sure you use the following writing guidelines to ensure consistency across articles:

- Do not use more than 550 characters per paragraph (about 90 words)
- For inline equations, use equations enclosed by two dollar signs, as in ```$\mathbf{F}=m\mathbf{a}$```. For equations displayed in the middle of the screen, with their own equation number, use double dollar signs followed by a new line and the equation, followed by a new line and again double dollar signs to close.
- Use a ```\tag{eq:<name>}``` in equations to allow for equation referencing. Use ```Eq.(\ref{eq:<name>})``` in the text to reference the equation tags. Use hyphens, not underscores, for the equation tags.

Example:

Newton's second law, as seen in Eq.(\ref{eq:newtons-second-law}), is given as:

```
$$
\mathbf{F}=m\mathbf{a} \tag{eq:newtons-second-law}
$$
```

The following would not work:

```
$$\mathbf{F}=m\mathbf{a} \tag{eq:newtons_second_law}$$
```

(Dollar signs are on the same line as the equation and the tag contains underscores)

After a pull request has been successfully merged, changes should automatically propagate to the website.