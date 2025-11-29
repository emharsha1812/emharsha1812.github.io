document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("paper-search");
  const cards = Array.from(document.querySelectorAll("#papershelf .papershelf-card"));
  const filterButtons = Array.from(document.querySelectorAll(".filter-btn"));

  if ((!input && filterButtons.length === 0) || cards.length === 0) {
    return;
  }

  const normalize = (text) => (text || "").toLowerCase();
  let activeTag = "all";

  const applyFilters = () => {
    const query = input ? normalize(input.value) : "";
    cards.forEach((card) => {
      const haystack = [card.dataset.title, card.dataset.authors, card.dataset.tags]
        .map(normalize)
        .join(" ");
      const matchesSearch = !query || haystack.includes(query);
      const tags = (card.dataset.tags || "").split(/\s+/).filter(Boolean);
      const matchesTag = activeTag === "all" || tags.includes(activeTag);
      card.style.display = matchesSearch && matchesTag ? "" : "none";
    });
  };

  if (input) {
    input.addEventListener("input", applyFilters);
  }

  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activeTag = button.dataset.tag;
      filterButtons.forEach((btn) => btn.classList.toggle("active", btn === button));
      applyFilters();
    });
  });

  applyFilters();
});
