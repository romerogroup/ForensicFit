document.addEventListener("DOMContentLoaded", function () {
    function onCopyButtonClick(event) {
        var preElement = event.target.closest(".highlight-python").querySelector("pre");
        var originalText = preElement.innerText;

        // Remove lines that don't start with ">>>"
        var modifiedText = originalText.split('\n').filter(line => line.startsWith(">>>") || line.startsWith("...")).join('\n');

        // Remove Python prompts (>>> and ...)
        modifiedText = modifiedText.replace(/^>>>(\s|$)/gm, '');

        // Remove leading and trailing white spaces
        modifiedText = modifiedText.trim();

        navigator.clipboard.writeText(modifiedText).then(function () {
            event.target.dataset.tooltip = "Copied";
            setTimeout(function () {
                event.target.dataset.tooltip = "Copy";
            }, 2000);
        }, function (err) {
            console.error('Failed to copy text: ', err);
        });

        event.preventDefault();
    }

    var copyButtons = document.querySelectorAll("button.copybtn");

    for (var i = 0; i < copyButtons.length; i++) {
        var copyButton = copyButtons[i];
        copyButton.addEventListener("click", onCopyButtonClick);
    }
});
