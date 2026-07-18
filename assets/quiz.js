(function () {
  document.querySelectorAll(".quiz[data-answer]").forEach((quiz) => {
    const answer = Number(quiz.dataset.answer);
    const feedback = quiz.querySelector(".fb");
    const buttons = Array.from(quiz.querySelectorAll(".opt"));

    buttons.forEach((button, index) => {
      button.addEventListener("click", () => {
        buttons.forEach((candidate, candidateIndex) => {
          candidate.disabled = true;
          if (candidateIndex === answer) candidate.classList.add("correct");
        });
        if (index !== answer) button.classList.add("wrong");
        feedback.classList.add("show");
      });
    });
  });
})();
