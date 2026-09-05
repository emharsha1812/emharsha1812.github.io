document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("blog-search");
  const input = document.getElementById("blog-search-input");
  const status = document.getElementById("blog-search-status");
  const results = document.getElementById("blog-search-results");
  const postList = document.getElementById("blog-post-list");
  const courses = document.getElementById("writing-courses");
  const pagination = document.getElementById("blog-pagination");
  const index = document.getElementById("blog-search-index");

  if (!form || !input || !status || !results || !postList || !index) return;

  let writings;
  try {
    writings = JSON.parse(index.textContent);
  } catch (error) {
    console.error("Unable to load the writing search index.", error);
    return;
  }

  const renderResults = (matches) => {
    results.replaceChildren();

    if (matches.length === 0) {
      const empty = document.createElement("li");
      empty.textContent = "No writings matched your search.";
      results.appendChild(empty);
      return;
    }

    matches.forEach((post) => {
      const item = document.createElement("li");
      const heading = document.createElement("h3");
      const link = document.createElement("a");
      const description = document.createElement("p");
      const meta = document.createElement("p");

      link.className = "post-title";
      link.href = post.url;
      link.textContent = post.title;
      heading.appendChild(link);
      item.appendChild(heading);

      if (post.description) {
        description.textContent = post.description;
        item.appendChild(description);
      }

      meta.className = "post-meta";
      meta.textContent = post.date;
      item.appendChild(meta);
      results.appendChild(item);
    });
  };

  const updateResults = () => {
    const query = input.value.trim().toLowerCase();

    if (!query) {
      postList.hidden = false;
      results.hidden = true;
      if (courses) courses.hidden = false;
      if (pagination) pagination.hidden = false;
      status.textContent = "";
      return;
    }

    const terms = query.split(/\s+/);
    const matches = writings.filter((post) => {
      const searchableText = `${post.title} ${post.description} ${post.content}`.toLowerCase();
      return terms.every((term) => searchableText.includes(term));
    });

    postList.hidden = true;
    results.hidden = false;
    if (courses) courses.hidden = true;
    if (pagination) pagination.hidden = true;
    status.textContent = `${matches.length} ${matches.length === 1 ? "writing" : "writings"} found`;
    renderResults(matches);
  };

  form.addEventListener("submit", (event) => event.preventDefault());
  input.addEventListener("input", updateResults);
});
