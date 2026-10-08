document.addEventListener("DOMContentLoaded", () => {
  const copyIcon = `
      <svg aria-hidden="true" viewBox="0 0 16 16" width="16" height="16">
        <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>
        <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
      </svg>
    `;

  const checkIcon = `
      <svg aria-hidden="true" viewBox="0 0 16 16" width="16" height="16">
        <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
      </svg>
    `;

  document.querySelectorAll("pre > code").forEach((codeBlock) => {
    const pre = codeBlock.parentElement;

    if (pre.querySelector(".copy-code-button")) {
      return;
    }

    const button = document.createElement("button");
    button.className = "copy-code-button";
    button.type = "button";
    button.innerHTML = copyIcon;
    button.setAttribute("aria-label", "Copy code");
    button.setAttribute("title", "Copy code");

    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(codeBlock.innerText);

        button.innerHTML = checkIcon;
        button.classList.add("copied");
        button.setAttribute("aria-label", "Copied");
        button.setAttribute("title", "Copied");

        setTimeout(() => {
          button.innerHTML = copyIcon;
          button.classList.remove("copied");
          button.setAttribute("aria-label", "Copy code");
          button.setAttribute("title", "Copy code");
        }, 1500);
      } catch {
        button.classList.add("failed");
        button.setAttribute("aria-label", "Copy failed");
        button.setAttribute("title", "Copy failed");

        setTimeout(() => {
          button.innerHTML = copyIcon;
          button.classList.remove("failed");
          button.setAttribute("aria-label", "Copy code");
          button.setAttribute("title", "Copy code");
        }, 1500);
      }
    });

    pre.classList.add("code-block-with-copy");
    pre.appendChild(button);
  });
});
