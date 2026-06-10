window.FaceMotion = {
  set(state, title, text) {
    const face = document.querySelector("#faceMotion");
    const titleNode = document.querySelector("#faceTitle");
    const textNode = document.querySelector("#faceText");
    if (!face || !titleNode || !textNode) {
      return;
    }
    face.dataset.state = state;
    titleNode.textContent = title;
    textNode.textContent = text;
  },
};
