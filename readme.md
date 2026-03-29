This repository contains the raw HTML of all articles hosted on [cfd.university](https://cfd.university/)

Having all articles on GitHub allows for collaborative improvements of articles. If you have found issues, inconsistencies, or errors with the articles, please feel free to open up a new issue.

Articles on [cfd.university](https://cfd.university) are pulled weekly (Tuesday at 8am GMT) from this repository, at which the cache will be flushed and rebuilt as well. Critical changes can be pushed quicker, but will require me to do that manually. 

The articles are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Contribution guideline

Depending on how comfortable you feel with git/GitHub and raw HTML, you can either:

1. Fork this repository and make changes directly to the raw HTML and submit a pull request.
2. Open an issue and state what needs fixing/changing. I can then make those changes myself. In this case, you don't need to touch the HTML yourself.

## Spelling rules

Use British English for all spelling/corrections, but feel free to ignore the following spellings (because I am proudly inconsistent):

- metre -> meter
- centre -> center
- aeroplane -> airplane

## Article structure

cfd.university was born on WordPress. WordPress does annotate the HTML and so you will find a lot of comments that have no influence now, as the website is now running using Python + Flask + Docker + :yellow_heart:.

Any changes made should continue to use this styling for consitency (though I have no plan to go back to WordPress. I'm glad I left that mess behind).

### Text changes

Text changes should be pretty straight forward. Edit the text you need. If you need a new paragraph, enclose it in a ```<p>``` tag as:

```html
<!-- wp:paragraph -->
<p>Your paragraph goes here</p>
<!-- /wp:paragraph -->
```

Keep the wordpress paragraph comments as well. Each paragraph cannot have more than 550 characters (excluding the ```<p></p>``` tags and comments).

### Equations

All equations displayed on cfd.university use standard [LaTeX syntax](https://www.overleaf.com/learn/latex/Mathematical_expressions). If you need to change equations inside a paragraph, make sure that you enclose an equation with ```[katex]``` and ```[/katex]```. For example, you may have a text passage like:

```html
Newton's second law states [katex]\mathbf{F}=m\mathbf{a}[/katex], from which we can derive the momentum equation.
```

If you want to make sure your equation is free of syntax errors, you can type them on [katex](https://katex.org/) directly and see how they render.

The ```[katex]``` and ```[/katex]``` tags are a left-over from WordPress. WordPress would scan each article before displaying it and convert all text inside these tags to equations using katex, an online equation rendering service. I have replicated that functionality so old HTML articles will render correctly. Therefore, we need to use this non-standard HTML syntax.

If you want to place the equation on a new line, you can enclose that with a ```div```:

```html
<!-- wp:katex/display-block -->
<div class="wp-block-katex-display-block katex-eq" data-katex-display="true"><pre>\mathbf{F}=m\mathbf{a}</pre></div>
<!-- /wp:katex/display-block -->
```

The ```div``` and ```pre``` tags need to be present, with the same classes and atributes. Internally, these will then be rendered with katex again.

Here are a few rules I use when typing equations to make them consistent:

- Scalars use a normal font face, vectors and matrices use a bold font face. As we can see in the example above, the force F and acceleration a are vectors, and so they get a bold font face using the ```\mathbf{}``` syntax (math bold face). The mass m is a scalar and is written without any styling.
- If you use parenthesis, start and end them with a leading ```\left``` and trailing ```\right``` identifier. This ensures parenthesis automatically scale to the correct size. For example, ```\left( \frac{1}{2} \right)```. Since we are using a fraction here, the standard parenthesis ```()``` would be too small, ```\left``` and ```\right``` scale them to the size of the fraction.
- When working with multiline equations, I almost always append a ```[1em]``` to the line break command, i.e. ```\\[1em]```. This gives just a bit of breathing room, especially when fractions are involved. I only deviate from that if the equations are rather compact. If in doubt, end a line in ```\\[1em]```. The exception here are matrices, for which I usally always just use a simple line break of ```\\```.
- When working with differentials (mostly in integrals, e.g. dx in an integral like int(x)dx), the differential is never italic but always straight. So, a differential is not ```dx``` but always ```\mathrm{d}x```, where ```\mathrm{}``` removes (math rm) the math styling. It works the same way as ```\text{d}x```, but I prefer the math version, i.e. ```\mathrm{}```.

### Code

Ah, now that is a bit of a challenge. I have used the [code block pro](https://code-block-pro.com/) plugin in WordPress, which I believe is one of the best features in the entire WordPress landscape! Unfortunately, it only works natively in WordPress, so we have to use a bot of a workaround.

1. Go to the [coding playground](https://code-block-pro.com/themes?theme=slack-dark&lang=cpp)
2. Paste in the code that you want to add (or probably modify).
3. Select the correct programming language from the right menu.
4. Select the correct styling from the left. cfd.university uses ```slack-dark```.
5. Once the code has been edited, right click on the code and go to the inspect menu.
6. You will have to find the parent ```div``` element, which start with ```<div class="leading-normal" style="opacity: 1;"><pre class="font-fira" style="background-color: transparent;">```. Copy the entire element, this is the HTML code that you can now paste into the HTML of the article.
7. This is probably one of the few, if not only element, that cannot be placed in a wordpress block, as the code block pro editor generates the HTML comments dynamically based on the code. So, whatever code you get by copying the HTML is what you can place in the article, no need to add further HTML comments.

Example, the following HTML was copied from the code block pro editor:

```html
<div class="leading-normal" style="opacity: 1;"><pre class="font-fira" style="background-color: transparent;"><span class="line"><span style="color: #C586C0">#include</span><span style="color: #569CD6"> </span><span style="color: #CE9178">&lt;iostream&gt;</span></span>
<span class="line"><span style="color: #569CD6">int</span><span style="color: #E6E6E6"> </span><span style="color: #DCDCAA">main</span><span style="color: #E6E6E6">() {</span></span>
<span class="line"><span style="color: #E6E6E6">   std::cout </span><span style="color: #D4D4D4">&lt;&lt;</span><span style="color: #E6E6E6"> </span><span style="color: #CE9178">"Hello cfd.university"</span><span style="color: #E6E6E6"> </span><span style="color: #D4D4D4">&lt;&lt;</span><span style="color: #E6E6E6"> std::endl;</span></span>
<span class="line"><span style="color: #E6E6E6">   </span><span style="color: #C586C0">return</span><span style="color: #E6E6E6"> </span><span style="color: #B5CEA8">0</span><span style="color: #E6E6E6">;</span></span>
<span class="line"><span style="color: #E6E6E6">}</span></span></pre></div>
```

This will render as follows:

<div class="leading-normal" style="opacity: 1;"><pre class="font-fira" style="background-color: transparent;"><span class="line"><span style="color: #C586C0">#include</span><span style="color: #569CD6"> </span><span style="color: #CE9178">&lt;iostream&gt;</span></span>
<span class="line"><span style="color: #569CD6">int</span><span style="color: #E6E6E6"> </span><span style="color: #DCDCAA">main</span><span style="color: #E6E6E6">() {</span></span>
<span class="line"><span style="color: #E6E6E6">   std::cout </span><span style="color: #D4D4D4">&lt;&lt;</span><span style="color: #E6E6E6"> </span><span style="color: #CE9178">"Hello cfd.university"</span><span style="color: #E6E6E6"> </span><span style="color: #D4D4D4">&lt;&lt;</span><span style="color: #E6E6E6"> std::endl;</span></span>
<span class="line"><span style="color: #E6E6E6">   </span><span style="color: #C586C0">return</span><span style="color: #E6E6E6"> </span><span style="color: #B5CEA8">0</span><span style="color: #E6E6E6">;</span></span>
<span class="line"><span style="color: #E6E6E6">}</span></span></pre></div>

### Section headings

Section headings can be modified but only to correct typos. Articles sometimes link to specific sections, so changing the link text would have adverse consequences. Don't change link text, even if they have typos. I can do that manually if need be.

### Acknolwedgement

Credit where credit is due! If you help to improve the articles, you will be listed at the bottom of the articles in the acknolwedgement section (it will only show up once at least one person is listed as a contributor).

If you prepare a pull request, please also change the meta data in the ```description.json``` file for that series/blog. For example, if you make changes to the first article in the ```07_10-key-concepts-everyone-must-understand-in-cfd``` series, then, modify the ```description.json``` file in ```07_10-key-concepts-everyone-must-understand-in-cfd/description.json```. Look for the article you have modified, for the first article, that would be:

```json
"01_how-to-derive-the-navier-stokes-equations-from-start-to-end.html": {
    "heading": "How to Derive the Navier-Stokes Equations: From start to end",
    "slug": "how-to-derive-the-navier-stokes-equations",
    "description": "If you want to know how to derive the Navier-Stokes equations, look no further; this article derives them from start to end, with explanations and no omissions!"
},
```

You will have to add a new entry called ```contributions```, which is a key value map (dictionary), where the key is your name (that will show up on the website) and the value is a link to your social media, personal website, etc. (try to avoid your only fans account *if possible*). An example is shown below:

```json
"01_how-to-derive-the-navier-stokes-equations-from-start-to-end.html": {
    "heading": "How to Derive the Navier-Stokes Equations: From start to end",
    "slug": "how-to-derive-the-navier-stokes-equations",
    "description": "If you want to know how to derive the Navier-Stokes equations, look no further; this article derives them from start to end, with explanations and no omissions!",
    "contributors": {
        "Tom-Robin Teschner": "https://github.com/tomrobin-teschner"
    }
},
```

If someone already contributed, simply add your name to the list:

```json
"01_how-to-derive-the-navier-stokes-equations-from-start-to-end.html": {
    "heading": "How to Derive the Navier-Stokes Equations: From start to end",
    "slug": "how-to-derive-the-navier-stokes-equations",
    "description": "If you want to know how to derive the Navier-Stokes equations, look no further; this article derives them from start to end, with explanations and no omissions!",
    "contributors": {
        "Tom-Robin Teschner": "https://github.com/tomrobin-teschner",
        "Milo Edwards": "https://www.youtube.com/@milo_edwards"
    }
},
```

Keep in mind this is ```json```, don't forget to add commas where necessary. If in doubt, throw the entire ```description.json``` file into a json validation tool like [jsonlint](https://jsonlint.com/) and see if it is valid before submitting a pull request. But, if you can do a pull request, you probably know how to handle JSON. Apologies for insulting your inteligence ...
