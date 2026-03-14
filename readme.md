This repository contains the raw HTML of all articles hosted on [cfd.university](https://cfd.university/)

Having all articles on GitHub allows for collaborative improvements of articles. If you have found issues, inconsistencies, or errors with the articles, please feel free to open up a new issue.

Articles on [cfd.university](https://cfd.university) are pulled weekly (Tuesday at 8am GMT) from this repository, at which the cache will be flushed and rebuild as well. Critical changes can be pushed quicker, but will require me to do that manually. 

The articles are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Contribution guideline

Depending on how comfortable you feel with raw HTML, you can either:

1. Make changes directly to the raw HTML and submit a pull request
2. Open an issue and state what needs fixing/changing. I can then make those changes myself. In this case, you don't need to touch the HTML yourself.

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

This is currently under development (well, that is fancy talk for saying I have it somewhere on my never ending todo list). I want to give acknowledgement to anyone contributing to improving the articles. For the moment, I will keep track of it manually, but the idea is that anyone contributing will appear on the website as well as a contributor, not jsut on github. Bear with me as I work through my todo list, it is coming.