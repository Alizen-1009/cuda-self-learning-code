(function () {
  function positionsFor(lane, operand) {
    const group = lane >> 2;
    const thread = lane & 3;
    const positions = [];

    if (operand === "a") {
      for (let index = 0; index < 8; index += 1) {
        const row = index < 2 || (index >= 4 && index < 6) ? group : group + 8;
        const col = thread * 2 + (index & 1) + (index >= 4 ? 8 : 0);
        positions.push({ row, col, register: `RA${index >> 1}`, element: `a${index}` });
      }
    } else if (operand === "b") {
      for (let index = 0; index < 4; index += 1) {
        const row = thread * 2 + (index & 1) + (index >= 2 ? 8 : 0);
        positions.push({ row, col: group, register: `RB${index >> 1}`, element: `b${index}` });
      }
    } else {
      for (let index = 0; index < 4; index += 1) {
        const row = index < 2 ? group : group + 8;
        const col = thread * 2 + (index & 1);
        positions.push({ row, col, register: `RC${index}`, element: `c${index}` });
      }
    }

    return positions;
  }

  function buildGrid(container, rows, cols) {
    container.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    const cells = new Map();
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const cell = document.createElement("span");
        cell.className = "matrix-cell";
        cell.textContent = `${row},${col}`;
        cell.title = `(${row}, ${col})`;
        container.appendChild(cell);
        cells.set(`${row},${col}`, cell);
      }
    }
    return cells;
  }

  document.querySelectorAll("[data-fragment-explorer]").forEach((root) => {
    const slider = root.querySelector("[data-lane-slider]");
    const badge = root.querySelector("[data-lane-badge]");
    const summary = root.querySelector("[data-fragment-summary]");
    const matrices = {
      a: buildGrid(root.querySelector('[data-matrix="a"]'), 16, 16),
      b: buildGrid(root.querySelector('[data-matrix="b"]'), 16, 8),
      c: buildGrid(root.querySelector('[data-matrix="c"]'), 16, 8),
    };

    function render() {
      const lane = Number(slider.value);
      badge.textContent = `lane ${lane}`;
      Object.values(matrices).forEach((cells) => {
        cells.forEach((cell) => cell.classList.remove("active", "pair"));
      });

      const lines = [];
      ["a", "b", "c"].forEach((operand) => {
        const positions = positionsFor(lane, operand);
        positions.forEach(({ row, col }) => matrices[operand].get(`${row},${col}`).classList.add("active"));
        if (operand !== "c") {
          for (let index = 0; index < positions.length; index += 2) {
            const pair = positions.slice(index, index + 2);
            pair.forEach(({ row, col }) => matrices[operand].get(`${row},${col}`).classList.add("pair"));
          }
        }
        lines.push(`${operand.toUpperCase()}: ${positions.map((p) => `${p.register}[${p.element}]=(${p.row},${p.col})`).join("  ")}`);
      });
      summary.textContent = lines.join("\n");
    }

    slider.addEventListener("input", render);
    render();
  });
})();
