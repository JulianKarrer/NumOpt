Reveal.initialize({
    katex: {
      local: './katex.ks',
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false }
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    plugins: [ RevealMath.KaTeX ]
  });
  